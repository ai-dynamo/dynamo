// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the connector's local axum control HTTP server.
//!
//! When `enabled = true` the connector leader exposes an axum router on
//! `bind_addr:port` that mirrors the trait-driven control API (reset,
//! register-leader, …). When `enabled = false` (the default) the connector
//! is reachable for control operations only via velo handlers — typically
//! proxied by `kvbm-hub`'s HTTP surface.
//!
//! Keeping the local axum off by default minimizes the number of fixed
//! ports operators have to plumb across vLLM/connector/engine processes.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Local axum control-plane configuration.
///
/// # JSON example
/// ```json
/// {
///   "enabled": true,
///   "bind_addr": "127.0.0.1",
///   "port": 9999
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ControlConfig {
    /// Whether to spawn the local axum control server. Default `false`:
    /// the connector relies on velo handlers and the hub's HTTP proxy.
    #[serde(default)]
    pub enabled: bool,

    /// Bind address for the local axum listener (default `0.0.0.0`).
    #[serde(default = "default_bind_addr")]
    pub bind_addr: String,

    /// Port for the local axum listener (default `9999`).
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ControlConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bind_addr: default_bind_addr(),
            port: default_port(),
        }
    }
}

fn default_bind_addr() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    9999
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_disabled() {
        let cfg = ControlConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.bind_addr, "0.0.0.0");
        assert_eq!(cfg.port, 9999);
    }

    #[test]
    fn deserialize_explicit_enabled() {
        let json = r#"{"enabled": true, "bind_addr": "127.0.0.1", "port": 19999}"#;
        let cfg: ControlConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.bind_addr, "127.0.0.1");
        assert_eq!(cfg.port, 19999);
    }

    #[test]
    fn deserialize_partial_uses_defaults() {
        // Only enabled present — bind_addr and port default.
        let json = r#"{"enabled": true}"#;
        let cfg: ControlConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.bind_addr, "0.0.0.0");
        assert_eq!(cfg.port, 9999);
    }

    #[test]
    fn deserialize_empty_uses_full_default() {
        let cfg: ControlConfig = serde_json::from_str("{}").unwrap();
        assert!(!cfg.enabled);
    }

    #[test]
    fn validate_ok() {
        assert!(ControlConfig::default().validate().is_ok());
    }
}
