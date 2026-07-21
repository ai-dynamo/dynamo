// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments and transport configuration for the SGLang sidecar.

use std::path::PathBuf;
use std::time::Duration;

/// Parsed sidecar arguments.
#[derive(clap::Parser, Debug, Clone)]
#[command(
    name = "dynamo-sglang-sidecar",
    about = "Dynamo sidecar for an out-of-process SGLang native gRPC server."
)]
pub struct Args {
    /// `host:port` (or URL) of SGLang's native `sglang.runtime.v1` service.
    #[arg(long, visible_alias = "grpc-endpoint", env = "SGLANG_GRPC_ENDPOINT")]
    pub sglang_endpoint: String,

    /// Number of independent HTTP/2 connections used for generation streams.
    #[arg(long, env = "SGLANG_GRPC_CONNECTIONS", default_value_t = 8)]
    pub sglang_connections: usize,

    /// Reachable host that decode workers use to connect to a prefill worker's
    /// SGLang disaggregation bootstrap port. By default this is derived from
    /// `dist_init_addr`, then a routable local address. This is required when
    /// discovery exposes only loopback or wildcard addresses.
    #[arg(long, env = "SGLANG_DISAGGREGATION_BOOTSTRAP_HOST")]
    pub bootstrap_host: Option<String>,

    /// Dynamo namespace for discovery routing.
    #[arg(long, env = "DYN_NAMESPACE", default_value = "dynamo")]
    pub namespace: String,

    /// Endpoint name exposed by this worker.
    #[arg(long, env = "DYN_ENDPOINT", default_value = "generate")]
    pub endpoint: String,

    /// Comma-separated endpoint types (for example `chat,completions`).
    #[arg(long, env = "DYN_ENDPOINT_TYPES", default_value = "chat,completions")]
    pub endpoint_types: String,

    /// Optional path to a custom Jinja chat template.
    #[arg(long, env = "DYN_CUSTOM_JINJA_TEMPLATE")]
    pub custom_jinja_template: Option<PathBuf>,

    /// Per-attempt connection timeout in seconds.
    #[arg(long, default_value_t = 30)]
    pub connect_timeout_secs: u64,

    /// Delay between connection/readiness attempts in seconds.
    #[arg(long, default_value_t = 2)]
    pub health_poll_interval_secs: u64,

    /// Total startup deadline in seconds.
    #[arg(long, default_value_t = 300)]
    pub health_deadline_secs: u64,
}

impl Args {
    pub fn transport(&self) -> TransportConfig {
        TransportConfig {
            connect_timeout: Duration::from_secs(self.connect_timeout_secs),
            poll_interval: Duration::from_secs(self.health_poll_interval_secs.max(1)),
            deadline: Duration::from_secs(self.health_deadline_secs),
            connections: self.sglang_connections.max(1),
        }
    }
}

/// Connection and readiness tunables shared by bootstrap and `start`.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub connect_timeout: Duration,
    pub poll_interval: Duration,
    pub deadline: Duration,
    pub connections: usize,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(30),
            poll_interval: Duration::from_secs(2),
            deadline: Duration::from_secs(300),
            connections: 1,
        }
    }
}

/// Normalize plaintext endpoint schemes for tonic.
///
/// TLS is intentionally rejected until the sidecar enables and tests tonic's
/// TLS transport feature instead of advertising an unusable `grpcs://` URL.
pub fn normalize_endpoint(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("SGLang gRPC endpoint must not be empty".to_string());
    }
    if let Some(rest) = trimmed.strip_prefix("grpc://") {
        if rest.is_empty() {
            return Err("SGLang gRPC endpoint host must not be empty".to_string());
        }
        Ok(format!("http://{rest}"))
    } else if trimmed.starts_with("grpcs://") || trimmed.starts_with("https://") {
        Err("TLS endpoints are not supported by the SGLang sidecar".to_string())
    } else if let Some(rest) = trimmed.strip_prefix("http://") {
        if rest.is_empty() {
            return Err("SGLang gRPC endpoint host must not be empty".to_string());
        }
        Ok(trimmed.to_string())
    } else if trimmed.contains("://") {
        Err(format!(
            "unsupported SGLang gRPC endpoint scheme: {trimmed}"
        ))
    } else {
        Ok(format!("http://{trimmed}"))
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_endpoint;

    #[test]
    fn normalizes_bare_and_grpc_endpoints() {
        assert_eq!(
            normalize_endpoint("127.0.0.1:30001").unwrap(),
            "http://127.0.0.1:30001",
        );
        assert_eq!(
            normalize_endpoint("grpc://host:7").unwrap(),
            "http://host:7"
        );
        assert!(normalize_endpoint("grpcs://host:8").is_err());
        assert!(normalize_endpoint(" https://host:9 ").is_err());
    }

    #[test]
    fn rejects_endpoints_without_hosts() {
        assert!(normalize_endpoint("http://").is_err());
        assert!(normalize_endpoint("grpc://").is_err());
    }
}
