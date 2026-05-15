// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for the leader control plane's optional (togglable) modules.
//!
//! The leader control plane is served over velo by the engine. `core` and
//! `transfer` are always on; `dev` and `test` are opt-in via this config.
//! When omitted, both default to `false`.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Leader control-plane module configuration.
///
/// # JSON example
/// ```json
/// {
///   "dev": true,
///   "test": false
/// }
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct ControlConfig {
    /// Enable the `dev` control module (`reset`). Off by default; safe to
    /// enable in production — no warning is logged when enabled.
    #[serde(default)]
    pub dev: bool,

    /// Enable the `test` control module (test-only block helpers). Off by
    /// default. May be enabled in production, but doing so logs a warning —
    /// its handlers are not intended for production use.
    #[serde(default)]
    pub test: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_disabled() {
        let cfg = ControlConfig::default();
        assert!(!cfg.dev);
        assert!(!cfg.test);
    }

    #[test]
    fn deserialize_explicit() {
        let cfg: ControlConfig = serde_json::from_str(r#"{"dev": true, "test": true}"#).unwrap();
        assert!(cfg.dev);
        assert!(cfg.test);
    }

    #[test]
    fn deserialize_partial_uses_defaults() {
        let cfg: ControlConfig = serde_json::from_str(r#"{"dev": true}"#).unwrap();
        assert!(cfg.dev);
        assert!(!cfg.test);
    }

    #[test]
    fn deserialize_empty_uses_full_default() {
        let cfg: ControlConfig = serde_json::from_str("{}").unwrap();
        assert!(!cfg.dev);
        assert!(!cfg.test);
    }

    #[test]
    fn validate_ok() {
        assert!(ControlConfig::default().validate().is_ok());
    }
}
