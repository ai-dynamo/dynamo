// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::time::Duration;

use clap::Args;
use dynamo_backend_common::CommonArgs;

const DEFAULT_GRPC_CONNECT_ATTEMPT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_GRPC_CONNECTIONS: NonZeroUsize = NonZeroUsize::new(8).unwrap();
const DEFAULT_GRPC_RETRY_INTERVAL_SECS: u64 = 1;
const DEFAULT_GRPC_STARTUP_DEADLINE_SECS: u64 = 300;

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

    /// Maximum total duration for sidecar gRPC startup.
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
mod tests {
    use std::time::Duration;

    use clap::Parser;

    use super::SidecarArgs;

    #[derive(Parser)]
    struct TestArgs {
        #[command(flatten)]
        sidecar: SidecarArgs,
    }

    #[test]
    fn parses_defaults_and_overrides() {
        let defaults = TestArgs::try_parse_from(["test"]).expect("parse defaults");
        let config = defaults.sidecar.grpc.config();
        assert_eq!(config.connections.get(), 8);
        assert_eq!(config.connect_attempt_timeout, Duration::from_secs(30));
        assert_eq!(config.retry_interval, Duration::from_secs(1));
        assert_eq!(config.startup_deadline, Duration::from_secs(300));

        let overrides = TestArgs::try_parse_from([
            "test",
            "--grpc-connections",
            "2",
            "--grpc-connect-attempt-timeout-secs",
            "7",
            "--grpc-retry-interval-secs",
            "3",
            "--grpc-startup-deadline-secs",
            "11",
        ])
        .expect("parse overrides");
        let config = overrides.sidecar.grpc.config();
        assert_eq!(config.connections.get(), 2);
        assert_eq!(config.connect_attempt_timeout, Duration::from_secs(7));
        assert_eq!(config.retry_interval, Duration::from_secs(3));
        assert_eq!(config.startup_deadline, Duration::from_secs(11));
    }

    #[test]
    fn rejects_zero_values() {
        for flag in [
            "--grpc-connections",
            "--grpc-connect-attempt-timeout-secs",
            "--grpc-retry-interval-secs",
            "--grpc-startup-deadline-secs",
        ] {
            assert!(TestArgs::try_parse_from(["test", flag, "0"]).is_err());
        }
    }
}
