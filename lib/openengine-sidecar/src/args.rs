// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;
use std::time::Duration;

#[derive(clap::Parser, Debug, Clone)]
#[command(
    name = "dynamo-openengine-sidecar",
    about = "Engine-neutral Dynamo worker for an out-of-process OpenEngine server"
)]
pub struct Args {
    /// OpenEngine gRPC endpoint. A bare host:port is interpreted as plaintext HTTP/2.
    #[arg(long, env = "OPENENGINE_ENDPOINT")]
    pub openengine_endpoint: String,

    /// Optional engine identity assertion (for example `tensorrt_llm`).
    #[arg(long, env = "OPENENGINE_EXPECTED_ENGINE")]
    pub expected_engine: Option<String>,

    /// Model to select when the OpenEngine server advertises more than one model.
    #[arg(long, env = "OPENENGINE_MODEL")]
    pub model: Option<String>,

    /// Number of independent HTTP/2 connections used for Generate streams.
    #[arg(long, env = "OPENENGINE_CONNECTIONS", default_value_t = 8)]
    pub openengine_connections: usize,

    #[arg(long, env = "DYN_NAMESPACE", default_value = "dynamo")]
    pub namespace: String,

    #[arg(long, env = "DYN_ENDPOINT", default_value = "generate")]
    pub endpoint: String,

    #[arg(long, env = "DYN_ENDPOINT_TYPES", default_value = "chat,completions")]
    pub endpoint_types: String,

    #[arg(long, env = "DYN_CUSTOM_JINJA_TEMPLATE")]
    pub custom_jinja_template: Option<PathBuf>,

    #[arg(long, default_value_t = 30)]
    pub connect_timeout_secs: u64,

    #[arg(long, default_value_t = 2)]
    pub health_poll_interval_secs: u64,

    #[arg(long, default_value_t = 300)]
    pub health_deadline_secs: u64,

    /// Maximum time to wait for OpenEngine's process-wide drain stream.
    #[arg(long, env = "OPENENGINE_DRAIN_TIMEOUT_SECS", default_value_t = 25)]
    pub drain_timeout_secs: u64,

    /// Polling interval for OpenEngine load snapshots.
    #[arg(long, default_value_t = 1)]
    pub load_poll_interval_secs: u64,
}

impl Args {
    pub fn transport(&self) -> TransportConfig {
        TransportConfig {
            connect_timeout: Duration::from_secs(self.connect_timeout_secs),
            poll_interval: Duration::from_secs(self.health_poll_interval_secs.max(1)),
            deadline: Duration::from_secs(self.health_deadline_secs),
            drain_timeout: Duration::from_secs(self.drain_timeout_secs.max(1)),
            load_poll_interval: Duration::from_secs(self.load_poll_interval_secs.max(1)),
            connections: self.openengine_connections.max(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub connect_timeout: Duration,
    pub poll_interval: Duration,
    pub deadline: Duration,
    pub drain_timeout: Duration,
    pub load_poll_interval: Duration,
    pub connections: usize,
}

pub fn normalize_endpoint(raw: &str) -> String {
    let value = raw.trim();
    if value.contains("://") {
        value.to_string()
    } else {
        format!("http://{value}")
    }
}
