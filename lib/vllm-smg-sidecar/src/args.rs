// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments for the vLLM SMG sidecar.

use std::path::PathBuf;
use std::time::Duration;

/// Parsed sidecar arguments.
#[derive(clap::Parser, Debug, Clone)]
#[command(
    name = "dynamo-vllm-smg-sidecar",
    about = "Dynamo vLLM SMG sidecar — drives upstream vLLM `serve --grpc` in aggregated text mode."
)]
pub struct Args {
    /// `host:port` (or full URL) of the upstream vLLM SMG gRPC server.
    ///
    /// A bare `host:port` is normalised to `http://host:port`.
    #[arg(long, env = "SMG_ENDPOINT")]
    pub smg_endpoint: String,

    /// Number of independent gRPC connections to open to the engine.
    #[arg(long, env = "SMG_CONNECTIONS", default_value_t = 8)]
    pub smg_connections: usize,

    /// Dynamo namespace for discovery routing.
    #[arg(long, env = "DYN_NAMESPACE", default_value = "dynamo")]
    pub namespace: String,

    /// Endpoint name exposed by this worker.
    #[arg(long, env = "DYN_ENDPOINT", default_value = "generate")]
    pub endpoint: String,

    /// Comma-separated endpoint types (e.g. `chat,completions`).
    #[arg(long, env = "DYN_ENDPOINT_TYPES", default_value = "chat,completions")]
    pub endpoint_types: String,

    /// Optional path to a custom Jinja chat template.
    #[arg(long, env = "DYN_CUSTOM_JINJA_TEMPLATE")]
    pub custom_jinja_template: Option<PathBuf>,

    /// Per-attempt connect timeout, in seconds, when dialling the engine.
    #[arg(long, default_value_t = 30)]
    pub connect_timeout_secs: u64,

    /// How often (seconds) to poll health while waiting for readiness.
    #[arg(long, default_value_t = 2)]
    pub health_poll_interval_secs: u64,

    /// Upper bound (seconds) on bootstrap connectivity and readiness.
    #[arg(long, default_value_t = 300)]
    pub health_deadline_secs: u64,
}

impl Args {
    /// Transport tunables distilled into [`Duration`]s.
    pub fn transport(&self) -> TransportConfig {
        TransportConfig {
            connect_timeout: Duration::from_secs(self.connect_timeout_secs),
            poll_interval: Duration::from_secs(self.health_poll_interval_secs.max(1)),
            deadline: Duration::from_secs(self.health_deadline_secs),
            connections: self.smg_connections.max(1),
        }
    }
}

/// Connection + readiness tunables shared by bootstrap and `start`.
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

/// Normalise an endpoint into a URI tonic accepts.
pub fn normalize_endpoint(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_host_port_gets_http_scheme() {
        assert_eq!(
            normalize_endpoint("127.0.0.1:50051"),
            "http://127.0.0.1:50051"
        );
    }

    #[test]
    fn explicit_scheme_is_preserved() {
        assert_eq!(normalize_endpoint("https://host:443"), "https://host:443");
        assert_eq!(normalize_endpoint("http://host:1"), "http://host:1");
    }

    #[test]
    fn whitespace_is_trimmed() {
        assert_eq!(normalize_endpoint("  host:9  "), "http://host:9");
    }
}
