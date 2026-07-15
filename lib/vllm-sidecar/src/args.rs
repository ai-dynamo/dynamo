// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments for the vLLM sidecar.
//!
//! The engine and model metadata are discovered from vLLM. Deployment topology
//! remains a Dynamo concern, so `--disaggregation-mode` is explicit.
//!
//! The remaining flags mirror the Dynamo runtime knobs from
//! [`dynamo_backend_common::CommonArgs`] (same names + env vars) **except**
//! `--component`, which is derived from the selected disaggregation mode.

use dynamo_backend_common::DisaggregationMode;
use std::path::PathBuf;
use std::time::Duration;

/// Parsed sidecar arguments.
#[derive(clap::Parser, Debug, Clone)]
#[command(
    name = "dynamo-vllm-sidecar",
    about = "Dynamo vLLM sidecar — drives an out-of-process vLLM engine over vLLM gRPC. \
             Model metadata is discovered; topology is configured explicitly."
)]
pub struct Args {
    /// `host:port` (or full URL) of the vLLM gRPC server.
    ///
    /// A bare `host:port` is normalised to `http://host:port`. This is the
    /// engine endpoint used by the sidecar.
    #[arg(long, env = "VLLM_GRPC_ENDPOINT")]
    pub grpc_endpoint: String,

    /// Dynamo topology role. vLLM deliberately does not expose deployment
    /// topology in its native protocol, so the sidecar owns this routing
    /// configuration explicitly.
    #[arg(
        long,
        env = "DYN_DISAGGREGATION_MODE",
        default_value_t = DisaggregationMode::Aggregated
    )]
    pub disaggregation_mode: DisaggregationMode,

    /// Number of independent gRPC connections to open to the engine.
    ///
    /// Streaming `generate` requests are round-robined across the pool so
    /// concurrent load is spread over multiple HTTP/2 connections rather than
    /// funneled through one connection's serialized frame processing (the
    /// overhead-bound throughput/stability bottleneck). This is a transport
    /// knob, not discovery — the value is not engine-reported.
    #[arg(long, env = "VLLM_GRPC_CONNECTIONS", default_value_t = 8)]
    pub grpc_connections: usize,

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

    /// How often (seconds) to poll `Health` while waiting for `READY`.
    #[arg(long, default_value_t = 2)]
    pub health_poll_interval_secs: u64,

    /// Upper bound (seconds) on how long to wait for the engine to become
    /// reachable (bootstrap) and to reach `READY` (start).
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
            connections: self.grpc_connections.max(1),
        }
    }
}

/// Connection + readiness tunables shared by bootstrap and `start`.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Per-attempt connect timeout when dialling the gRPC server.
    pub connect_timeout: Duration,
    /// Delay between reconnect / `Health` poll attempts.
    pub poll_interval: Duration,
    /// Total budget for "become reachable" / "become ready".
    pub deadline: Duration,
    /// Size of the `generate` connection pool opened in `start()`. Bootstrap
    /// discovery always uses a single throwaway connection regardless.
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
///
/// A bare `host:port` gets an `http://` scheme. TLS is not configured for this
/// private pod-local control channel, so other schemes are rejected.
pub fn normalize_endpoint(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.starts_with("http://") {
        Ok(trimmed.to_string())
    } else if trimmed.contains("://") {
        Err(format!(
            "unsupported vLLM gRPC endpoint scheme in `{trimmed}`; use pod-local http://"
        ))
    } else {
        Ok(format!("http://{trimmed}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_host_port_gets_http_scheme() {
        assert_eq!(
            normalize_endpoint("127.0.0.1:50051").unwrap(),
            "http://127.0.0.1:50051"
        );
    }

    #[test]
    fn only_plain_http_scheme_is_supported() {
        assert!(normalize_endpoint("https://host:443").is_err());
        assert!(normalize_endpoint("grpc://host:443").is_err());
        assert_eq!(
            normalize_endpoint("http://host:1").unwrap(),
            "http://host:1"
        );
    }

    #[test]
    fn whitespace_is_trimmed() {
        assert_eq!(normalize_endpoint("  host:9  ").unwrap(), "http://host:9");
    }
}
