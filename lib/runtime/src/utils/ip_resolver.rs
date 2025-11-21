// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/// # Arguments
/// * `resolver` - An implementation of IpResolver trait for getting local IP addresses
///
/// # Returns
/// A string representation of the resolved IP address
pub fn get_http_rpc_host_with_resolver<R: IpResolver>(resolver: R) -> String {
    let resolved_ip = resolver.local_ip().or_else(|err| match err {
        Error::LocalIpAddressNotFound => resolver.local_ipv6(),
        _ => Err(err),
    });

    match resolved_ip {
        Ok(addr) => addr,
        Err(Error::LocalIpAddressNotFound) => IpAddr::from([127, 0, 0, 1]),
        Err(_) => IpAddr::from([127, 0, 0, 1]), // Fallback for any other error
    }
    .to_string()
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
        assert_eq!(result, "2001:db8::1");
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

        // Should be a valid IP address
        let _: IpAddr = result.parse().expect("Should be a valid IP address");
    }
}
