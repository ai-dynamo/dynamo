// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Debug configuration for KVBM.
//!
//! Controls recording and other debug-time features.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Debug configuration.
///
/// # V1 Compatibility
///
/// - `DYN_KVBM_ENABLE_RECORD` → `recording`
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct DebugConfig {
    /// Enable KVBM recording for debugging and replay.
    /// Default: false
    #[serde(default)]
    pub recording: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let config = DebugConfig::default();
        assert!(!config.recording);
    }

    #[test]
    fn test_serde_roundtrip() {
        let json = r#"{"recording": true}"#;
        let config: DebugConfig = serde_json::from_str(json).unwrap();
        assert!(config.recording);

        let serialized = serde_json::to_string(&config).unwrap();
        let roundtrip: DebugConfig = serde_json::from_str(&serialized).unwrap();
        assert!(roundtrip.recording);
    }

    #[test]
    fn test_empty_json_uses_default() {
        let json = r#"{}"#;
        let config: DebugConfig = serde_json::from_str(json).unwrap();
        assert!(!config.recording);
    }
}
