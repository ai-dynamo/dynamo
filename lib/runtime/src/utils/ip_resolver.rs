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

/// Format a host and port into a valid socket address string
///
/// This function handles both IPv4 and IPv6 addresses correctly.
/// IPv6 addresses are wrapped in brackets as required by the socket address syntax.
///
/// # Arguments
/// * `host` - The host address (IPv4, IPv6, or hostname)
/// * `port` - The port number
///
/// # Returns
/// A properly formatted socket address string
///
/// # Examples
/// ```
/// use dynamo_runtime::utils::format_socket_addr;
///
/// assert_eq!(format_socket_addr("192.168.1.1", 8080), "192.168.1.1:8080");
/// assert_eq!(format_socket_addr("::1", 8080), "[::1]:8080");
/// assert_eq!(format_socket_addr("2001:db8::1", 9999), "[2001:db8::1]:9999");
/// assert_eq!(format_socket_addr("localhost", 3000), "localhost:3000");
/// ```
pub fn format_socket_addr(host: &str, port: u16) -> String {
    // Check if the host is an IPv6 address (contains colons but is not already bracketed)
    if host.contains(':') && !host.starts_with('[') {
        format!("[{}]:{}", host, port)
    } else {
        format!("{}:{}", host, port)
    }
}

/// Format a host and port with an optional path suffix
///
/// Similar to `format_socket_addr` but allows appending a path.
/// Useful for TCP endpoints that include routing information.
///
/// # Arguments
/// * `host` - The host address (IPv4, IPv6, or hostname)
/// * `port` - The port number
/// * `path` - Optional path to append (e.g., "/endpoint_name")
///
/// # Returns
/// A properly formatted address string with optional path
pub fn format_socket_addr_with_path(host: &str, port: u16, path: &str) -> String {
    let base = format_socket_addr(host, port);
    if path.is_empty() {
        base
    } else if path.starts_with('/') {
        format!("{}{}", base, path)
    } else {
        format!("{}/{}", base, path)
    }
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

    #[test]
    fn test_format_socket_addr_ipv4() {
        assert_eq!(format_socket_addr("192.168.1.1", 8080), "192.168.1.1:8080");
        assert_eq!(format_socket_addr("0.0.0.0", 9999), "0.0.0.0:9999");
        assert_eq!(format_socket_addr("127.0.0.1", 3000), "127.0.0.1:3000");
    }

    #[test]
    fn test_format_socket_addr_ipv6() {
        assert_eq!(format_socket_addr("::1", 8080), "[::1]:8080");
        assert_eq!(
            format_socket_addr("2001:db8::1", 9999),
            "[2001:db8::1]:9999"
        );
        assert_eq!(
            format_socket_addr("fe80::1%eth0", 3000),
            "[fe80::1%eth0]:3000"
        );
        assert_eq!(format_socket_addr("::", 8080), "[::]:8080");
    }

    #[test]
    fn test_format_socket_addr_already_bracketed() {
        // Should not double-bracket
        assert_eq!(format_socket_addr("[::1]", 8080), "[::1]:8080");
    }

    #[test]
    fn test_format_socket_addr_hostname() {
        assert_eq!(format_socket_addr("localhost", 8080), "localhost:8080");
        assert_eq!(
            format_socket_addr("my-service.local", 9999),
            "my-service.local:9999"
        );
    }

    #[test]
    fn test_format_socket_addr_with_path_ipv4() {
        assert_eq!(
            format_socket_addr_with_path("192.168.1.1", 8080, "endpoint"),
            "192.168.1.1:8080/endpoint"
        );
        assert_eq!(
            format_socket_addr_with_path("192.168.1.1", 8080, "/endpoint"),
            "192.168.1.1:8080/endpoint"
        );
        assert_eq!(
            format_socket_addr_with_path("192.168.1.1", 8080, ""),
            "192.168.1.1:8080"
        );
    }

    #[test]
    fn test_format_socket_addr_with_path_ipv6() {
        assert_eq!(
            format_socket_addr_with_path("::1", 9999, "generate"),
            "[::1]:9999/generate"
        );
        assert_eq!(
            format_socket_addr_with_path("2001:db8::1", 9999, "/api/v1"),
            "[2001:db8::1]:9999/api/v1"
        );
    }

    #[test]
    fn test_format_socket_addr_parseable() {
        use std::net::SocketAddr;

        // IPv4 addresses should be parseable
        let addr: SocketAddr = format_socket_addr("192.168.1.1", 8080).parse().unwrap();
        assert_eq!(addr.port(), 8080);

        // IPv6 addresses should be parseable
        let addr: SocketAddr = format_socket_addr("::1", 9999).parse().unwrap();
        assert_eq!(addr.port(), 9999);

        let addr: SocketAddr = format_socket_addr("2001:db8::1", 3000).parse().unwrap();
        assert_eq!(addr.port(), 3000);
    }
}
