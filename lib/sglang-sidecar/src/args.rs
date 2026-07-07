// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments and transport configuration for the SGLang gRPC sidecar.

use std::path::PathBuf;
use std::time::Duration;

/// Parsed sidecar arguments.
#[derive(clap::Parser, Debug, Clone)]
#[command(
    name = "dynamo-sglang-sidecar",
    about = "Dynamo SGLang sidecar — drives an out-of-process SGLang native gRPC server."
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
    /// `dist_init_addr`, then from the gRPC endpoint host.
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

/// Normalize endpoint schemes for tonic. `grpc://` and `grpcs://` are common
/// user-facing spellings but tonic expects `http://` and `https://`.
pub fn normalize_endpoint(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("grpc://") {
        format!("http://{rest}")
    } else if let Some(rest) = trimmed.strip_prefix("grpcs://") {
        format!("https://{rest}")
    } else if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_endpoint;

    #[test]
    fn normalizes_bare_and_grpc_endpoints() {
        assert_eq!(
            normalize_endpoint("127.0.0.1:30001"),
            "http://127.0.0.1:30001"
        );
        assert_eq!(normalize_endpoint("grpc://host:7"), "http://host:7");
        assert_eq!(normalize_endpoint("grpcs://host:8"), "https://host:8");
        assert_eq!(normalize_endpoint(" https://host:9 "), "https://host:9");
    }
}
