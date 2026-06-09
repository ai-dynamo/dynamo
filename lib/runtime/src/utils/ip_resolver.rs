// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local IP address resolution for advertising endpoints.

use crate::pipeline::network::tcp::server::{DefaultIpResolver, IpResolver};
use local_ip_address::{Error, list_afinet_netifas};
use std::net::{IpAddr, Ipv4Addr};

const FALLBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// Resolve the local IP for advertising endpoints, falling back to 127.0.0.1.
///
/// IPv6 addresses are bracketed (e.g. `[::1]`) so the result is safe to
/// interpolate into a `host:port` URL.
pub fn local_ip_for_advertise() -> String {
    resolve(DefaultIpResolver)
}

/// TCP RPC host: `DYN_TCP_RPC_HOST` if set to an IP assigned in this network
/// namespace, otherwise the resolved local IP.
pub fn tcp_rpc_host_from_env() -> String {
    rpc_host_from_env("DYN_TCP_RPC_HOST", local_ip_for_advertise)
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

fn parse_env_host_ip(host: &str) -> Option<IpAddr> {
    let host = host.trim();
    if host.is_empty() {
        return None;
    }
    let host = host
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host);
    host.parse().ok()
}

fn current_interface_ips() -> Option<Vec<IpAddr>> {
    match list_afinet_netifas() {
        Ok(interfaces) => Some(interfaces.into_iter().map(|(_, ip)| ip).collect()),
        Err(err) => {
            tracing::warn!(
                %err,
                "Could not list local interfaces while validating RPC host env var"
            );
            None
        }
    }
}

fn select_rpc_host_from_value<F>(
    env_var: &str,
    configured_host: String,
    local_ips: Option<&[IpAddr]>,
    fallback: F,
) -> String
where
    F: FnOnce() -> String,
{
    if configured_host.trim().is_empty() {
        let resolved_host = fallback();
        tracing::warn!(
            env_var = %env_var,
            resolved_host = %resolved_host,
            "Ignoring empty RPC host env var"
        );
        return resolved_host;
    }

    let Some(configured_ip) = parse_env_host_ip(&configured_host) else {
        return configured_host;
    };

    if let Some(local_ips) = local_ips {
        if !local_ips.contains(&configured_ip) {
            let resolved_host = fallback();
            tracing::warn!(
                env_var = %env_var,
                configured_host = %configured_host,
                resolved_host = %resolved_host,
                "Ignoring RPC host env var because it is not assigned to this network namespace"
            );
            return resolved_host;
        }
    }

    configured_host
}

fn rpc_host_from_env<F>(env_var: &str, fallback: F) -> String
where
    F: FnOnce() -> String,
{
    let Ok(configured_host) = std::env::var(env_var) else {
        return fallback();
    };
    let local_ips = current_interface_ips();
    select_rpc_host_from_value(env_var, configured_host, local_ips.as_deref(), fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn both_fail_uses_localhost() {
        let r = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
        };
        assert_eq!(resolve(r), "127.0.0.1");
    }

    #[test]
    fn stale_rpc_host_env_ip_falls_back_to_resolved_ip() {
        let result = select_rpc_host_from_value(
            "DYN_TCP_RPC_HOST",
            "10.0.0.1".to_string(),
            Some(&[IpAddr::from([10, 0, 0, 2])]),
            || "10.0.0.2".to_string(),
        );
        assert_eq!(result, "10.0.0.2");
    }

    #[test]
    fn current_rpc_host_env_ip_is_used() {
        let result = select_rpc_host_from_value(
            "DYN_TCP_RPC_HOST",
            "10.0.0.2".to_string(),
            Some(&[IpAddr::from([10, 0, 0, 2])]),
            || "10.0.0.3".to_string(),
        );
        assert_eq!(result, "10.0.0.2");
    }

    #[test]
    fn rpc_host_env_hostname_is_used() {
        let result = select_rpc_host_from_value(
            "DYN_TCP_RPC_HOST",
            "worker-0".to_string(),
            Some(&[IpAddr::from([10, 0, 0, 2])]),
            || "10.0.0.2".to_string(),
        );
        assert_eq!(result, "worker-0");
    }

    #[test]
    fn empty_rpc_host_env_falls_back() {
        let result = select_rpc_host_from_value(
            "DYN_TCP_RPC_HOST",
            " ".to_string(),
            Some(&[IpAddr::from([10, 0, 0, 2])]),
            || "10.0.0.2".to_string(),
        );
        assert_eq!(result, "10.0.0.2");
    }
}
