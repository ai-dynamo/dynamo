// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use dynamo_backend_common::DynamoError;

use crate::invalid_argument;

/// Validated plaintext gRPC endpoint containing only a scheme and authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrpcEndpoint(String);

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
        if parsed.host().is_none() {
            return Err(invalid_argument(format!(
                "`{argument}` must include a host"
            )));
        }
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
        Ok(Self(format!("http://{authority}")))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GrpcEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::GrpcEndpoint;

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
    }

    #[test]
    fn rejects_unsupported_or_ambiguous_endpoints() {
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
        ] {
            assert!(GrpcEndpoint::parse(endpoint, ARGUMENT).is_err());
        }
    }
}
