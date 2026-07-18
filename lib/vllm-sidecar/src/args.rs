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

    /// Direct HTTP admin endpoint of the same vLLM engine. Required when
    /// `DYN_ENABLE_RL=true` so Dynamo can publish it through `/v1/rl/workers`.
    #[arg(long, env = "VLLM_HTTP_ENDPOINT")]
    pub admin_endpoint: Option<String>,

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
    normalize_pod_local_http_endpoint(raw, EndpointPath::AuthorityOnly)
        .map_err(|_| invalid_grpc_endpoint(raw))
}

pub fn normalize_admin_endpoint(raw: &str) -> Result<String, String> {
    normalize_pod_local_http_endpoint(raw, EndpointPath::Admin)
        .map_err(|_| invalid_admin_endpoint(raw))
}

#[derive(Clone, Copy)]
enum EndpointPath {
    AuthorityOnly,
    Admin,
}

fn normalize_pod_local_http_endpoint(raw: &str, endpoint_path: EndpointPath) -> Result<String, ()> {
    let trimmed = raw.trim();
    let candidate = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let parsed = url::Url::parse(&candidate).map_err(|_| ())?;
    let authority_start = candidate.find("://").ok_or(())? + 3;
    let authority_end = candidate[authority_start..]
        .find(['/', '?', '#'])
        .map_or(candidate.len(), |offset| authority_start + offset);
    let explicit_port = candidate[authority_start..authority_end]
        .rsplit_once(':')
        .and_then(|(_, port)| port.parse::<u16>().ok())
        .filter(|port| *port != 0)
        .ok_or(())?;
    let valid_path = match endpoint_path {
        EndpointPath::AuthorityOnly => matches!(parsed.path(), "" | "/"),
        EndpointPath::Admin => matches!(parsed.path(), "" | "/" | "/v1" | "/v1/"),
    };
    if parsed.scheme() != "http"
        || parsed.host().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || !valid_path
    {
        return Err(());
    }
    let origin = parsed.origin().ascii_serialization();
    Ok(if parsed.port().is_some() {
        origin
    } else {
        format!("{origin}:{explicit_port}")
    })
}

fn invalid_grpc_endpoint(raw: &str) -> String {
    format!("invalid vLLM gRPC endpoint `{raw}`; expected pod-local http://host:port with no path")
}

fn invalid_admin_endpoint(raw: &str) -> String {
    format!(
        "invalid vLLM admin endpoint `{raw}`; expected pod-local http://host:port with optional /v1"
    )
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
        assert_eq!(
            normalize_endpoint("http://host:80").unwrap(),
            "http://host:80"
        );
    }

    #[test]
    fn whitespace_is_trimmed() {
        assert_eq!(normalize_endpoint("  host:9  ").unwrap(), "http://host:9");
    }

    #[test]
    fn grpc_endpoint_requires_an_authority_with_explicit_port_only() {
        for invalid in [
            "host",
            ":50051",
            "host:port",
            "host :50051",
            "host:0",
            "http://user@host:50051",
            "http://host:50051/v1",
            "http://host:50051?query=1",
            "http://host:50051#fragment",
        ] {
            assert!(normalize_endpoint(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn admin_endpoint_uses_the_same_pod_local_http_validation() {
        assert_eq!(
            normalize_admin_endpoint("  127.0.0.1:8120/v1  ").unwrap(),
            "http://127.0.0.1:8120"
        );
        assert!(normalize_admin_endpoint("https://worker:8120").is_err());
        assert!(normalize_admin_endpoint("worker").is_err());
        assert!(normalize_admin_endpoint(":8120").is_err());
        assert!(normalize_admin_endpoint("worker:port").is_err());
        assert!(normalize_admin_endpoint("worker :8120").is_err());
        assert!(normalize_admin_endpoint("http://user@worker:8120").is_err());
        assert!(normalize_admin_endpoint("http://worker:8120/path").is_err());
        assert!(normalize_admin_endpoint("http://worker:8120?query=1").is_err());
    }
}
