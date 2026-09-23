// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::time::Duration;

use clap::Args;
use dynamo_backend_common::CommonArgs;

use crate::GrpcEndpoint;

const DEFAULT_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_GRPC_CONNECTIONS: NonZeroUsize = NonZeroUsize::new(8).unwrap();
const DEFAULT_GRPC_RETRY_INTERVAL_SECS: u64 = 1;
const DEFAULT_GRPC_STARTUP_DEADLINE_SECS: u64 = 1800;

fn parse_grpc_endpoint(raw: &str) -> Result<GrpcEndpoint, String> {
    GrpcEndpoint::parse(raw, "--grpc-endpoint").map_err(|error| error.to_string())
}

/// CLI flags shared by gRPC sidecars.
#[derive(Args, Clone, Debug)]
pub struct GrpcTransportArgs {
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

    /// Maximum duration for each sidecar gRPC startup phase.
    #[arg(
        long = "grpc-startup-deadline-secs",
        env = "DYN_SIDECAR_GRPC_STARTUP_DEADLINE_SECS",
        default_value_t = DEFAULT_GRPC_STARTUP_DEADLINE_SECS,
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub grpc_startup_deadline_secs: u64,
}

impl GrpcTransportArgs {
    pub fn config(&self) -> GrpcTransportConfig {
        GrpcTransportConfig {
            connections: self.grpc_connections,
            connect_attempt_timeout: Duration::from_secs(self.grpc_connect_attempt_timeout_secs),
            retry_interval: Duration::from_secs(self.grpc_retry_interval_secs),
            startup_deadline: Duration::from_secs(self.grpc_startup_deadline_secs),
        }
    }
}

/// Standard worker and gRPC transport flags for a sidecar executable.
#[derive(Args, Clone, Debug)]
pub struct SidecarArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Engine gRPC endpoint as `host:port` or a plaintext gRPC/HTTP URL.
    #[arg(
        long,
        env = "DYN_SIDECAR_GRPC_ENDPOINT",
        value_parser = parse_grpc_endpoint
    )]
    pub grpc_endpoint: GrpcEndpoint,

    #[command(flatten)]
    pub grpc: GrpcTransportArgs,
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
mod unit_common_args {
    sidecar_shared_tests!(args);
}
