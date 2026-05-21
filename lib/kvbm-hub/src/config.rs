// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;

use figment::Figment;
use figment::providers::{Env, Format, Json, Serialized, Toml};
use serde::{Deserialize, Serialize};

use crate::protocol;
use crate::protocol::PrimaryConfig;

/// Hub server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubConfig {
    /// Address to bind both listeners to (default `0.0.0.0`).
    pub bind_addr: IpAddr,
    /// Discovery HTTP port (default `1337`).
    pub discovery_port: u16,
    /// Control-plane HTTP port (default `8337`).
    pub control_port: u16,
    /// Liveness TTL in seconds for the in-memory registry (default `30`).
    /// Ignored when a custom registry backend is injected.
    #[serde(default = "default_registration_ttl_secs")]
    pub registration_ttl_secs: u64,
    /// Reaper tick interval in seconds for the in-memory registry
    /// (default `10`). Ignored when a custom registry backend is injected.
    #[serde(default = "default_prune_interval_secs")]
    pub prune_interval_secs: u64,
    /// Optional velo transport port. When set, the hub binds a
    /// `TcpTransport` to `bind_addr:velo_port` and participates in velo
    /// active messaging. When `None` (default), the hub runs
    /// discovery-only.
    #[serde(default)]
    pub velo_port: Option<u16>,
    /// Period (seconds) of the hub-driven heartbeat probe loop. Each
    /// tick, the hub velo-pushes a `HeartbeatRequest` to every
    /// registered instance and refreshes the registry on ack. Ignored
    /// when `velo_port` is `None` (no velo, no probes). Default `10`.
    #[serde(default = "default_heartbeat_interval_secs")]
    pub heartbeat_interval_secs: u64,
    /// Number of consecutive probe failures before the hub unregisters
    /// an instance. Default `3`. With the default 10s interval this
    /// removes a hung instance after ~30s — well under the 30s
    /// registry TTL so the reaper rarely sees stale entries when velo
    /// is in use.
    #[serde(default = "default_heartbeat_max_failures")]
    pub heartbeat_max_failures: u32,
    /// Hub-wide shared config ("primary"). Must-match fields here are validated
    /// against every registrant; advisory fields seed generated connector
    /// config. Feature configs (e.g. `indexer`) inherit unset sizing from
    /// here. See [`PrimaryConfig`].
    #[serde(default)]
    pub primary: PrimaryConfig,
    /// Optional KV indexer feature. When set, the hub binds a ZMQ `SUB`
    /// ingest socket and serves the index under `/v1/features/indexer`.
    /// `None` (default) leaves the feature off.
    #[serde(default)]
    pub indexer: Option<IndexerConfig>,
}

/// Configuration for the optional KV indexer feature.
///
/// Sizing (`max_seq_len` / `block_size`) is optional and inherits from
/// [`HubConfig::primary`] when unset — they are the same must-match values, so
/// operators set them once on `primary`. The binary resolves the effective
/// sizing before constructing the [`IndexerManager`](crate::IndexerManager).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexerConfig {
    /// Maximum sequence length (tokens). Must be evenly divisible by
    /// `block_size`; the index is presized to `max_seq_len / block_size`
    /// position buckets. Inherits from `primary.max_seq_len` when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<usize>,
    /// Block size (tokens per block). Must match the publishers' page size.
    /// Inherits from `primary.block_size` when `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
    /// ZMQ bind spec for the ingest `SUB` socket. Default `tcp://0.0.0.0:0`
    /// (OS-assigned port, reported via `GET /config`).
    #[serde(default)]
    pub zmq_bind: Option<String>,
    /// Host advertised to publishers in `GET /config`'s `zmq_endpoint`.
    /// Default `127.0.0.1`; multi-host deployments must set this to a
    /// routable address.
    #[serde(default)]
    pub advertise_host: Option<String>,
}

fn default_registration_ttl_secs() -> u64 {
    30
}
fn default_prune_interval_secs() -> u64 {
    10
}
fn default_heartbeat_interval_secs() -> u64 {
    10
}
fn default_heartbeat_max_failures() -> u32 {
    3
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            bind_addr: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            discovery_port: protocol::DEFAULT_DISCOVERY_PORT,
            control_port: protocol::DEFAULT_CONTROL_PORT,
            registration_ttl_secs: default_registration_ttl_secs(),
            prune_interval_secs: default_prune_interval_secs(),
            velo_port: None,
            heartbeat_interval_secs: default_heartbeat_interval_secs(),
            heartbeat_max_failures: default_heartbeat_max_failures(),
            primary: PrimaryConfig::default(),
            indexer: None,
        }
    }
}

impl HubConfig {
    /// Builds a [`Figment`] with priority: defaults → optional config file → `KVBM_HUB_*` env vars.
    ///
    /// The caller (binary) merges CLI arg overrides on top of the returned Figment.
    /// `KVBM_HUB_CONFIG` is excluded from the env layer — it is consumed by the CLI
    /// before this method is called.
    pub fn figment(config_path: Option<&Path>) -> Figment {
        let mut f = Figment::new().merge(Serialized::defaults(HubConfig::default()));
        if let Some(path) = config_path {
            if path.extension().is_some_and(|e| e == "json") {
                f = f.merge(Json::file(path));
            } else {
                f = f.merge(Toml::file(path));
            }
        }
        f.merge(Env::prefixed("KVBM_HUB_").ignore(&["CONFIG"]))
    }
}
