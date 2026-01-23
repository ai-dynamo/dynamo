// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! IP resolution utilities for getting local IP addresses with fallback support

use crate::pipeline::network::tcp::server::{DefaultIpResolver, IpResolver};
use local_ip_address::Error;
use std::net::IpAddr;

/// Get the local IP address for HTTP RPC host binding, using IpResolver with fallback to 127.0.0.1
///
/// This function attempts to resolve the local IP address using the provided resolver.
/// If resolution fails, it falls back to 127.0.0.1 (localhost).
///
/// IPv6 addresses are wrapped with brackets for safe URL construction (e.g., `[::1]`).
///
/// # Arguments
/// * `resolver` - An implementation of IpResolver trait for getting local IP addresses
///
/// # Returns
/// A string representation of the resolved IP address (IPv6 addresses are bracketed)
pub fn get_http_rpc_host_with_resolver<R: IpResolver>(resolver: R) -> String {
    let resolved_ip = resolver.local_ip().or_else(|err| match err {
        Error::LocalIpAddressNotFound => resolver.local_ipv6(),
        _ => Err(err),
    });

    let addr = match resolved_ip {
        Ok(addr) => addr,
        Err(Error::LocalIpAddressNotFound) => IpAddr::from([127, 0, 0, 1]),
        Err(_) => IpAddr::from([127, 0, 0, 1]), // Fallback for any other error
    };

    // Wrap IPv6 addresses with brackets for safe URL construction
    // e.g., "2001:db8::1" becomes "[2001:db8::1]" so that "{host}:{port}" is valid
    match addr {
        IpAddr::V6(_) => format!("[{}]", addr),
        IpAddr::V4(_) => addr.to_string(),
    }
}

/// Get the local IP address for HTTP RPC host binding using the default resolver
///
/// This is a convenience function that uses the DefaultIpResolver.
/// It follows the same logic as the TcpStreamServer for IP resolution.
///
/// # Returns
/// A string representation of the resolved IP address, with fallback to "127.0.0.1"
pub fn get_http_rpc_host() -> String {
    get_http_rpc_host_with_resolver(DefaultIpResolver)
}

/// Get the HTTP RPC host from environment variable or resolve local IP as fallback
///
/// This function checks the DYN_HTTP_RPC_HOST environment variable first.
/// If not set, it uses IP resolution to determine the local IP address.
///
/// # Returns
/// A string representation of the HTTP RPC host address
pub fn get_http_rpc_host_from_env() -> String {
    std::env::var("DYN_HTTP_RPC_HOST").unwrap_or_else(|_| get_http_rpc_host())
}

/// Get the TCP RPC host from environment variable or resolve local IP as fallback
///
/// This function checks the DYN_TCP_RPC_HOST environment variable first.
/// If not set, it uses IP resolution to determine the local IP address.
///
/// # Returns
/// A string representation of the TCP RPC host address
pub fn get_tcp_rpc_host_from_env() -> String {
    std::env::var("DYN_TCP_RPC_HOST").unwrap_or_else(|_| get_http_rpc_host())
}

#[cfg(test)]
mod tests {
    use super::*;
    use local_ip_address::Error;

    // Mock resolver for testing
    struct MockIpResolver {
        ipv4_result: Result<IpAddr, Error>,
        ipv6_result: Result<IpAddr, Error>,
    }

    impl IpResolver for MockIpResolver {
        fn local_ip(&self) -> Result<IpAddr, Error> {
            match &self.ipv4_result {
                Ok(addr) => Ok(*addr),
                Err(Error::LocalIpAddressNotFound) => Err(Error::LocalIpAddressNotFound),
                Err(_) => Err(Error::LocalIpAddressNotFound), // Simplify for testing
            }
        }

        fn local_ipv6(&self) -> Result<IpAddr, Error> {
            match &self.ipv6_result {
                Ok(addr) => Ok(*addr),
                Err(Error::LocalIpAddressNotFound) => Err(Error::LocalIpAddressNotFound),
                Err(_) => Err(Error::LocalIpAddressNotFound), // Simplify for testing
            }
        }
    }

    #[test]
    fn test_get_http_rpc_host_with_successful_ipv4() {
        let resolver = MockIpResolver {
            ipv4_result: Ok(IpAddr::from([192, 168, 1, 100])),
            ipv6_result: Ok(IpAddr::from([0, 0, 0, 0, 0, 0, 0, 1])),
        };

        let result = get_http_rpc_host_with_resolver(resolver);
        assert_eq!(result, "192.168.1.100");
    }

    #[test]
    fn test_get_http_rpc_host_with_ipv4_fail_ipv6_success() {
        let resolver = MockIpResolver {
            ipv4_result: Err(Error::LocalIpAddressNotFound),
            ipv6_result: Ok(IpAddr::from([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1])),
        };

        let result = get_http_rpc_host_with_resolver(resolver);
        // IPv6 addresses should be bracketed for safe URL construction
        assert_eq!(result, "[2001:db8::1]");
    }

    #[test]
    fn test_get_http_rpc_host_with_both_fail() {
        let resolver = MockIpResolver {
            ipv4_result: Err(Error::LocalIpAddressNotFound),
            ipv6_result: Err(Error::LocalIpAddressNotFound),
        };

        let result = get_http_rpc_host_with_resolver(resolver);
        assert_eq!(result, "127.0.0.1");
    }

    #[test]
    fn test_get_http_rpc_host_from_env_with_env_var() {
        // Set environment variable
        unsafe {
            std::env::set_var("DYN_HTTP_RPC_HOST", "10.0.0.1");
        }

        let result = get_http_rpc_host_from_env();
        assert_eq!(result, "10.0.0.1");

        // Clean up
        unsafe {
            std::env::remove_var("DYN_HTTP_RPC_HOST");
        }
    }

    #[test]
    fn test_get_http_rpc_host_from_env_without_env_var() {
        // Note: We can't reliably unset environment variables in tests
        // This test assumes DYN_HTTP_RPC_HOST is not set to a specific test value

        let result = get_http_rpc_host_from_env();
        // Should return some IP address (either resolved or fallback)
        assert!(!result.is_empty());

        // Should be a valid IP address (strip brackets for IPv6 before parsing)
        let ip_str = result.trim_start_matches('[').trim_end_matches(']');
        let _: IpAddr = ip_str.parse().expect("Should be a valid IP address");
    }

    #[test]
    fn test_ipv6_address_is_bracketed() {
        let resolver = MockIpResolver {
            ipv4_result: Err(Error::LocalIpAddressNotFound),
            ipv6_result: Ok(IpAddr::from([0xfd00, 0xdead, 0xbeef, 0, 0, 0, 0, 2])),
        };

        let result = get_http_rpc_host_with_resolver(resolver);
        // IPv6 must be bracketed for URLs like http://{host}:{port}/path
        assert!(result.starts_with('['), "IPv6 should start with '['");
        assert!(result.ends_with(']'), "IPv6 should end with ']'");
        assert_eq!(result, "[fd00:dead:beef::2]");
    }

    #[test]
    fn test_ipv4_address_not_bracketed() {
        let resolver = MockIpResolver {
            ipv4_result: Ok(IpAddr::from([10, 0, 0, 1])),
            ipv6_result: Err(Error::LocalIpAddressNotFound),
        };

        let result = get_http_rpc_host_with_resolver(resolver);
        // IPv4 should NOT be bracketed
        assert!(!result.contains('['), "IPv4 should not contain '['");
        assert_eq!(result, "10.0.0.1");
    }
}
