// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;

use figment::Figment;
use figment::providers::{Env, Format, Json, Serialized, Toml};
use serde::{Deserialize, Serialize};

use crate::protocol;

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
}

fn default_registration_ttl_secs() -> u64 {
    30
}
fn default_prune_interval_secs() -> u64 {
    10
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
