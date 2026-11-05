// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use dynamo_backend_common::DynamoError;

use crate::invalid_argument;

/// The host a Unix-socket engine is reached at for any other port it serves.
/// The socket is on this host, and the engine shares this pod's network, so
/// loopback reaches it. IPv4, because an engine listening on `0.0.0.0` does not
/// accept on `::1`.
const UNIX_SOCKET_HOST: &str = "127.0.0.1";

/// Validated plaintext HTTP endpoint derived from a sidecar's gRPC host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpEndpoint {
    endpoint: url::Url,
}

impl HttpEndpoint {
    /// Parse an HTTP endpoint, allowing a path prefix but no credentials, query, or fragment.
    pub fn parse(raw: &str, argument: &str) -> Result<Self, DynamoError> {
        let endpoint = raw.trim();
        if endpoint.is_empty() {
            return Err(invalid_argument(format!(
                "`{argument}` is required and must specify an HTTP server address"
            )));
        }
        if endpoint
            .split_once("://")
            .is_some_and(|(_, authority)| authority.starts_with('/'))
        {
            return Err(invalid_argument(format!(
                "`{argument}` must include a host"
            )));
        }
        let endpoint = url::Url::parse(endpoint).map_err(|error| {
            invalid_argument(format!("invalid HTTP endpoint for `{argument}`: {error}"))
        })?;
        if !matches!(endpoint.scheme(), "http" | "https") || endpoint.host().is_none() {
            return Err(invalid_argument(format!(
                "`{argument}` must use HTTP or HTTPS and include a host"
            )));
        }
        if !endpoint.username().is_empty() || endpoint.password().is_some() {
            return Err(invalid_argument(format!(
                "`{argument}` must not include user information"
            )));
        }
        if endpoint.query().is_some() || endpoint.fragment().is_some() {
            return Err(invalid_argument(format!(
                "`{argument}` must not include a query or fragment"
            )));
        }
        Ok(Self { endpoint })
    }

    /// Reuse the host from a validated gRPC endpoint with a discovered HTTP port.
    /// A Unix-socket endpoint's host is loopback; see [`GrpcEndpoint::authority_host`].
    pub fn from_grpc(grpc_endpoint: &GrpcEndpoint, port: u16) -> Result<Self, DynamoError> {
        if port == 0 {
            return Err(invalid_argument("HTTP endpoint port must not be zero"));
        }
        let host = format!("http://{}", grpc_endpoint.authority_host());
        let mut endpoint = url::Url::parse(&host).map_err(|error| {
            invalid_argument(format!(
                "could not derive HTTP endpoint from `{grpc_endpoint}`: {error}"
            ))
        })?;
        endpoint.set_port(Some(port)).map_err(|()| {
            invalid_argument(format!(
                "could not set HTTP port {port} on `{grpc_endpoint}`"
            ))
        })?;
        Ok(Self { endpoint })
    }

    /// Return this endpoint with `path`, preserving its validated authority.
    pub fn with_path(&self, path: &str) -> url::Url {
        let mut endpoint = self.endpoint.clone();
        endpoint.set_path(path);
        endpoint
    }

    pub fn as_str(&self) -> &str {
        self.endpoint.as_str()
    }
}

impl fmt::Display for HttpEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Validated plaintext gRPC endpoint: a scheme and authority, or a Unix socket
/// named by absolute path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrpcEndpoint {
    /// `http://HOST:PORT`, or `unix:///PATH` for a Unix socket.
    endpoint: String,
    authority_host: String,
}

impl GrpcEndpoint {
    pub fn parse(raw: &str, argument: &str) -> Result<Self, DynamoError> {
        let endpoint = raw.trim();
        if endpoint.is_empty() {
            return Err(invalid_argument(format!(
                "`{argument}` is required and must specify a gRPC server address, such as `http://HOST:PORT`"
            )));
        }

        if let Some(target) = endpoint.strip_prefix("unix:") {
            return Self::parse_unix_socket(target, argument);
        }
        let normalized = if let Some(authority) = endpoint.strip_prefix("grpc://") {
            format!("http://{authority}")
        } else if endpoint.starts_with("http://") {
            endpoint.to_string()
        } else if endpoint.starts_with("grpcs://") || endpoint.starts_with("https://") {
            return Err(invalid_argument(format!(
                "TLS endpoints are not supported by `{argument}`"
            )));
        } else if endpoint.contains("://") {
            return Err(invalid_argument(format!(
                "unsupported endpoint scheme in `{argument}`: `{endpoint}`"
            )));
        } else {
            format!("http://{endpoint}")
        };

        let parsed = url::Url::parse(&normalized).map_err(|error| {
            invalid_argument(format!("invalid gRPC endpoint for `{argument}`: {error}"))
        })?;
        let authority_host = match parsed.host() {
            Some(url::Host::Domain(host)) => host.to_string(),
            Some(url::Host::Ipv4(host)) => host.to_string(),
            Some(url::Host::Ipv6(host)) => format!("[{host}]"),
            None => {
                return Err(invalid_argument(format!(
                    "`{argument}` must include a host"
                )));
            }
        };
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(invalid_argument(format!(
                "`{argument}` must not include user information"
            )));
        }
        if parsed.path() != "/" || parsed.query().is_some() || parsed.fragment().is_some() {
            return Err(invalid_argument(format!(
                "`{argument}` must contain only a plaintext scheme and authority"
            )));
        }

        let authority = &parsed[url::Position::BeforeHost..url::Position::AfterPort];
        Ok(Self {
            endpoint: format!("http://{authority}"),
            authority_host,
        })
    }

    /// `target` is what follows `unix:`. Only `unix:///ABSOLUTE/PATH` is
    /// accepted: `unix:///run/engine.sock` names `/run/engine.sock`.
    fn parse_unix_socket(target: &str, argument: &str) -> Result<Self, DynamoError> {
        let path = target
            .strip_prefix("//")
            .filter(|path| path.starts_with('/') && !path.ends_with('/'))
            .ok_or_else(|| {
                invalid_argument(format!(
                    "`{argument}` must name a Unix socket file by absolute path, as \
                     `unix:///ABSOLUTE/PATH`, got `unix:{target}`"
                ))
            })?;
        // The socket need not exist yet, but a path no socket address can hold
        // would otherwise fail every connection attempt until the deadline.
        std::os::unix::net::SocketAddr::from_pathname(path).map_err(|error| {
            invalid_argument(format!(
                "`{argument}` names an unusable Unix socket path `{path}`: {error}"
            ))
        })?;
        Ok(Self {
            endpoint: format!("unix://{path}"),
            authority_host: UNIX_SOCKET_HOST.to_string(),
        })
    }

    pub fn as_str(&self) -> &str {
        &self.endpoint
    }

    /// Host formatted for use in a URI authority, including IPv6 brackets. For
    /// a Unix socket, the loopback address that reaches the engine's other ports.
    pub fn authority_host(&self) -> &str {
        &self.authority_host
    }

    pub(crate) fn is_unix_socket(&self) -> bool {
        self.endpoint.starts_with("unix:")
    }
}

impl fmt::Display for GrpcEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.endpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::{GrpcEndpoint, HttpEndpoint};

    const ARGUMENT: &str = "--test-endpoint";

    #[test]
    fn normalizes_plaintext_endpoints() {
        assert_eq!(
            GrpcEndpoint::parse(" 127.0.0.1:50051 ", ARGUMENT)
                .unwrap()
                .as_str(),
            "http://127.0.0.1:50051"
        );
        assert_eq!(
            GrpcEndpoint::parse("http://server:50051", ARGUMENT)
                .unwrap()
                .as_str(),
            "http://server:50051"
        );
        assert_eq!(
            GrpcEndpoint::parse("grpc://server:50051", ARGUMENT)
                .unwrap()
                .as_str(),
            "http://server:50051"
        );
        let ipv6 = GrpcEndpoint::parse("http://[2001:db8::1]:50051", ARGUMENT).unwrap();
        assert_eq!(ipv6.as_str(), "http://[2001:db8::1]:50051");
        assert_eq!(ipv6.authority_host(), "[2001:db8::1]");
        assert_eq!(
            GrpcEndpoint::parse(" unix:///run/engine/engine.sock ", ARGUMENT)
                .unwrap()
                .as_str(),
            "unix:///run/engine/engine.sock"
        );
    }

    #[test]
    fn rejects_unsupported_or_ambiguous_endpoints() {
        let too_long = format!("unix:///{}", "a".repeat(200));
        for endpoint in [
            "",
            " ",
            "http://",
            "grpc://",
            "https://server",
            "other://server",
            "http://user:password@server:50051",
            "http://server:50051/path",
            "http://server:50051?token=secret",
            "http://server:50051#fragment",
            "unix:/run/engine.sock",
            "unix://localhost/run/engine.sock",
            "unix:///run/engine/",
            too_long.as_str(),
        ] {
            assert!(GrpcEndpoint::parse(endpoint, ARGUMENT).is_err());
        }
    }

    #[test]
    fn derives_http_endpoint_from_grpc_host() {
        let grpc = GrpcEndpoint::parse("http://server:30001", ARGUMENT).unwrap();
        let http = HttpEndpoint::from_grpc(&grpc, 30000).unwrap();
        assert_eq!(http.as_str(), "http://server:30000/");
        assert_eq!(
            http.with_path("/generate").as_str(),
            "http://server:30000/generate"
        );

        let grpc = GrpcEndpoint::parse("http://[2001:db8::1]:30001", ARGUMENT).unwrap();
        let http = HttpEndpoint::from_grpc(&grpc, 30000).unwrap();
        assert_eq!(http.as_str(), "http://[2001:db8::1]:30000/");
        assert!(HttpEndpoint::from_grpc(&grpc, 0).is_err());

        let grpc = GrpcEndpoint::parse("unix:///run/engine/engine.sock", ARGUMENT).unwrap();
        let http = HttpEndpoint::from_grpc(&grpc, 30000).unwrap();
        assert_eq!(http.as_str(), "http://127.0.0.1:30000/");
    }

    #[test]
    fn parses_http_endpoints_with_path_prefixes() {
        for endpoint in [
            "http://worker:8120",
            "https://worker.example.com",
            "https://worker.example.com/admin/v1",
        ] {
            assert!(HttpEndpoint::parse(endpoint, "--http-endpoint").is_ok());
        }
    }

    #[test]
    fn rejects_invalid_http_endpoints() {
        for endpoint in [
            "",
            "worker:8120",
            "grpc://worker:8120",
            "https:///admin",
            "HTTP:///admin",
            "https://user:token@worker.example.com/admin",
            "https://worker.example.com/admin?token=secret",
            "https://worker.example.com/admin#fragment",
        ] {
            assert!(HttpEndpoint::parse(endpoint, "--http-endpoint").is_err());
        }
    }
}
