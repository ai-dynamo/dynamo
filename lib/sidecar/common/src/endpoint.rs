// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use dynamo_backend_common::DynamoError;

use crate::invalid_argument;

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
    pub fn from_grpc(grpc_endpoint: &GrpcEndpoint, port: u16) -> Result<Self, DynamoError> {
        if port == 0 {
            return Err(invalid_argument("HTTP endpoint port must not be zero"));
        }
        let mut endpoint = url::Url::parse(grpc_endpoint.as_str()).map_err(|error| {
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

/// Validated plaintext gRPC endpoint containing only a scheme and authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrpcEndpoint {
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

    pub fn as_str(&self) -> &str {
        &self.endpoint
    }

    /// Host formatted for use in a URI authority, including IPv6 brackets.
    pub fn authority_host(&self) -> &str {
        &self.authority_host
    }
}

impl fmt::Display for GrpcEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.endpoint)
    }
}

#[cfg(test)]
#[path = "../../testkit/tests/unit/common/endpoint.rs"]
mod unit_endpoint;
