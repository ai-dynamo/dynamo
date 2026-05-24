// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! clap `Args` for the kvbm_service binary. Kept separate so the library
//! crate doesn't bake in CLI behavior.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use clap::Args;

use super::ServiceConfigCli;

#[derive(Debug, Args, Default)]
pub struct ServiceFlags {
    /// HTTP sidecar listen address (env: KVBM_SERVICE_HTTP_ADDR).
    #[arg(long, env = "KVBM_SERVICE_HTTP_ADDR")]
    pub http_addr: Option<SocketAddr>,

    /// Explicit UDS path. If unset, derived from --uds-dir + pid + http port.
    #[arg(long, env = "KVBM_SERVICE_UDS_PATH")]
    pub uds_path: Option<PathBuf>,

    /// Parent directory for the auto-derived UDS path.
    #[arg(long, env = "KVBM_SERVICE_UDS_DIR")]
    pub uds_dir: Option<PathBuf>,

    /// Graceful shutdown deadline. Accepts humantime strings ("60s", "90s",
    /// "5m"). Default: wait indefinitely. Minimum 60s when set.
    #[arg(
        long,
        env = "KVBM_SERVICE_SHUTDOWN_GRACE",
        value_parser = humantime::parse_duration,
    )]
    pub shutdown_grace: Option<Duration>,

    /// Optional TOML config file. Settings layered under env and CLI.
    #[arg(long, env = "KVBM_SERVICE_CONFIG")]
    pub config: Option<PathBuf>,
}

impl ServiceFlags {
    pub fn to_cli(&self) -> ServiceConfigCli {
        ServiceConfigCli {
            http_addr: self.http_addr,
            uds_path: self.uds_path.clone(),
            uds_dir: self.uds_dir.clone(),
            shutdown_grace_ms: self
                .shutdown_grace
                .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
        }
    }

    pub fn config_path(&self) -> Option<&std::path::Path> {
        self.config.as_deref()
    }
}
