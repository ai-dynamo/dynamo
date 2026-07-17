// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common CLI flags every Rust backend needs.
//!
//! Flag names match the Python runtime's `WorkerConfig` knobs so operators see
//! the same CLI surface across Rust and Python backends. Engines extend this
//! with their own `clap` `Args` using `#[command(flatten)]`.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

use clap::Args;

use crate::disagg::DisaggregationMode;

const DEFAULT_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_GRPC_CONNECTIONS: NonZeroUsize = NonZeroUsize::new(8).unwrap();
const DEFAULT_GRPC_RETRY_INTERVAL_SECS: u64 = 1;
const DEFAULT_GRPC_STARTUP_DEADLINE_SECS: u64 = 300;

/// Normalize a plaintext gRPC client endpoint for tonic.
///
/// Bare addresses and `grpc://` URLs are translated to `http://`. TLS and
/// other schemes are rejected until the caller configures tonic TLS.
pub fn normalize_grpc_endpoint(raw: &str, argument: &str) -> Result<String, String> {
    let endpoint = raw.trim();
    if endpoint.is_empty() {
        return Err(format!(
            "`{argument}` is required and must specify a gRPC server address, such as `http://HOST:PORT`"
        ));
    }

    let normalized = if let Some(authority) = endpoint.strip_prefix("grpc://") {
        format!("http://{authority}")
    } else if endpoint.starts_with("http://") {
        endpoint.to_string()
    } else if endpoint.starts_with("grpcs://") || endpoint.starts_with("https://") {
        return Err(format!("TLS endpoints are not supported by `{argument}`"));
    } else if endpoint.contains("://") {
        return Err(format!(
            "unsupported endpoint scheme in `{argument}`: `{endpoint}`"
        ));
    } else {
        format!("http://{endpoint}")
    };

    let parsed = url::Url::parse(&normalized)
        .map_err(|error| format!("invalid gRPC endpoint for `{argument}`: {error}"))?;
    if parsed.host().is_none() {
        return Err(format!("`{argument}` must include a host"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!("`{argument}` must not include user information"));
    }
    if parsed.path() != "/" || parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(format!(
            "`{argument}` must contain only a plaintext scheme and authority"
        ));
    }

    let authority = &parsed[url::Position::BeforeHost..url::Position::AfterPort];
    Ok(format!("http://{authority}"))
}

/// CLI flags that every Rust backend needs. Flatten these into your engine's
/// `Args` struct:
///
/// ```ignore
/// #[derive(clap::Parser)]
/// struct EngineArgs {
///     #[clap(flatten)]
///     common: CommonArgs,
///
///     #[arg(long, default_value = "sample-model")]
///     model_name: String,
/// }
/// ```
#[derive(Args, Clone, Debug)]
pub struct CommonArgs {
    /// Dynamo namespace for discovery routing.
    #[arg(long, default_value = "dynamo", env = "DYN_NAMESPACE")]
    pub namespace: String,

    /// Component name within the namespace.
    #[arg(long, default_value = "backend", env = "DYN_COMPONENT")]
    pub component: String,

    /// Endpoint name exposed by this worker.
    #[arg(long, default_value = "generate", env = "DYN_ENDPOINT")]
    pub endpoint: String,

    /// Comma-separated list of supported endpoint types
    /// (`chat`, `completions`, `embeddings`, ...).
    #[arg(long, default_value = "chat,completions", env = "DYN_ENDPOINT_TYPES")]
    pub endpoint_types: String,

    /// Optional path to a custom Jinja chat template, used instead of the
    /// template shipped with the model.
    #[arg(long, env = "DYN_CUSTOM_JINJA_TEMPLATE")]
    pub custom_jinja_template: Option<PathBuf>,

    /// Dynamo frontend tool-call parser name for this model.
    #[arg(long = "dyn-tool-call-parser", env = "DYN_TOOL_CALL_PARSER")]
    pub dyn_tool_call_parser: Option<String>,

    /// Dynamo frontend reasoning parser name for this model.
    #[arg(long = "dyn-reasoning-parser", env = "DYN_REASONING_PARSER")]
    pub dyn_reasoning_parser: Option<String>,

    /// Exclude tools from the chat template when tool_choice is none.
    #[arg(
        long = "exclude-tools-when-tool-choice-none",
        env = "DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub exclude_tools_when_tool_choice_none: bool,

    /// Disaggregation role: `agg` (default), `prefill`, `decode`, or `encode`.
    /// Prefill workers register with `ModelType::empty()` and
    /// `WorkerType::Prefill` regardless of `endpoint_types`; decode and encode
    /// workers do not advertise a local KV indexer. Encode workers register as
    /// `WorkerType::Encode` and are not exposed on the public chat/completions
    /// surface.
    #[arg(
        long,
        value_enum,
        default_value_t = DisaggregationMode::Aggregated,
        env = "DYN_DISAGGREGATION_MODE",
    )]
    pub disaggregation_mode: DisaggregationMode,

    /// Declare an upstream Encode peer in this worker's topology `needs`.
    /// Meaningful only on `--disaggregation-mode agg` and `prefill`.
    /// Setting it on `decode` or `encode` is rejected at startup.
    ///
    /// Scope: Rust backends consuming `CommonArgs` via clap. Python
    /// backends populate this from their own runtime config -- the Python
    /// shim does not read this env var.
    #[arg(long, default_value_t = false, env = "DYN_ROUTE_TO_ENCODER")]
    pub route_to_encoder: bool,

    /// Number of parallel sidecar gRPC connections. The default of eight was
    /// sufficient in sidecar load tests to avoid connection-level throttling
    /// at high request concurrency.
    #[arg(
        long = "grpc-connections",
        env = "DYN_SIDECAR_GRPC_CONNECTIONS",
        default_value_t = DEFAULT_GRPC_CONNECTIONS
    )]
    pub grpc_connections: NonZeroUsize,

    /// Maximum duration of one sidecar gRPC connection attempt.
    #[arg(
        long = "grpc-connect-attempt-timeout-secs",
        env = "DYN_SIDECAR_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS",
        default_value_t = DEFAULT_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub grpc_connect_attempt_timeout_secs: u64,

    /// Delay between sidecar gRPC connection attempts.
    #[arg(
        long = "grpc-retry-interval-secs",
        env = "DYN_SIDECAR_GRPC_RETRY_INTERVAL_SECS",
        default_value_t = DEFAULT_GRPC_RETRY_INTERVAL_SECS,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub grpc_retry_interval_secs: u64,

    /// Maximum total duration for sidecar gRPC startup.
    #[arg(
        long = "grpc-startup-deadline-secs",
        env = "DYN_SIDECAR_GRPC_STARTUP_DEADLINE_SECS",
        default_value_t = DEFAULT_GRPC_STARTUP_DEADLINE_SECS,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub grpc_startup_deadline_secs: u64,
}

impl CommonArgs {
    pub fn grpc_transport_config(&self) -> GrpcTransportConfig {
        GrpcTransportConfig {
            connections: self.grpc_connections,
            connect_attempt_timeout: Duration::from_secs(self.grpc_connect_attempt_timeout_secs),
            retry_interval: Duration::from_secs(self.grpc_retry_interval_secs),
            startup_deadline: Duration::from_secs(self.grpc_startup_deadline_secs),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GrpcTransportConfig {
    pub connections: NonZeroUsize,
    pub connect_attempt_timeout: Duration,
    pub retry_interval: Duration,
    pub startup_deadline: Duration,
}

impl Default for GrpcTransportConfig {
    fn default() -> Self {
        Self {
            connections: DEFAULT_GRPC_CONNECTIONS,
            connect_attempt_timeout: Duration::from_secs(DEFAULT_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS),
            retry_interval: Duration::from_secs(DEFAULT_GRPC_RETRY_INTERVAL_SECS),
            startup_deadline: Duration::from_secs(DEFAULT_GRPC_STARTUP_DEADLINE_SECS),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_grpc_endpoint;

    const ARGUMENT: &str = "--test-endpoint";

    #[test]
    fn normalizes_plaintext_grpc_endpoints() {
        assert_eq!(
            normalize_grpc_endpoint(" 127.0.0.1:50051 ", ARGUMENT).unwrap(),
            "http://127.0.0.1:50051"
        );
        assert_eq!(
            normalize_grpc_endpoint("http://server:50051", ARGUMENT).unwrap(),
            "http://server:50051"
        );
        assert_eq!(
            normalize_grpc_endpoint("grpc://server:50051", ARGUMENT).unwrap(),
            "http://server:50051"
        );
    }

    #[test]
    fn rejects_missing_hosts_and_unsupported_schemes() {
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
            assert!(normalize_grpc_endpoint(endpoint, ARGUMENT).is_err());
        }
    }
}
