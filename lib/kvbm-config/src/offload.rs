// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offload policy configuration for KVBM.
//!
//! Defines configuration for offload policies that control which blocks
//! are transferred between storage tiers (G1→G2, G2→G3).
//!
//! # Policy Types
//!
//! - `pass_all`: No filtering, all blocks pass
//! - `presence`: Skip blocks already present in destination tier
//! - `presence_lfu`: Presence check + LFU count threshold
//!
//! # Configuration
//!
//! Policies are configured per tier transition. Multiple policies in the
//! `policies` list are applied in order with implicit AND logic (all must pass).
//!
//! ## JSON Example
//!
//! ```json
//! {
//!   "offload": {
//!     "g1_to_g2": {
//!       "policies": ["presence"],
//!       "presence": {}
//!     },
//!     "g2_to_g3": {
//!       "policies": ["presence_lfu"],
//!       "presence_lfu": { "min_lfu_count": 8 }
//!     }
//!   }
//! }
//! ```

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Policy type enum for serialization.
///
/// Each variant corresponds to a policy implementation in the kvbm crate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyType {
    /// PassAllPolicy - no filtering, all blocks pass
    PassAll,
    /// PresenceFilter - skip blocks already in destination tier
    Presence,
    /// PresenceAndLFUFilter - presence check + LFU threshold
    PresenceLfu,
}

/// Configuration for presence filter.
///
/// Currently has no parameters, but the struct exists for future extensibility
/// and to maintain consistent configuration patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct PresenceFilterConfig {}

/// Default LFU count threshold.
fn default_min_lfu_count() -> u32 {
    8
}

/// Configuration for presence + LFU filter.
///
/// Combines presence checking with LFU (Least Frequently Used) count threshold.
/// Only blocks with access count above the threshold are offloaded.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PresenceLfuFilterConfig {
    /// Minimum LFU count threshold for offload.
    ///
    /// Blocks must have been accessed more than this many times to be
    /// considered for offload. This prevents offloading rarely-used blocks.
    ///
    /// Default: 8
    #[serde(default = "default_min_lfu_count")]
    #[validate(range(min = 1))]
    pub min_lfu_count: u32,
}

impl Default for PresenceLfuFilterConfig {
    fn default() -> Self {
        Self {
            min_lfu_count: default_min_lfu_count(),
        }
    }
}

/// Configuration for a tier transition (e.g., G1→G2, G2→G3).
///
/// Defines which policies to apply when offloading blocks between tiers.
/// Policies are evaluated in order with implicit AND logic - a block must
/// pass ALL policies to be transferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct TierOffloadConfig {
    /// Ordered list of policies to apply (implicit AND).
    ///
    /// If empty, tier-specific defaults are applied by the engine.
    /// Policies are evaluated in order; a block must pass all to be transferred.
    #[serde(default)]
    pub policies: Vec<PolicyType>,

    /// Presence filter configuration.
    ///
    /// Used when "presence" is in the policies list.
    #[serde(default)]
    #[validate(nested)]
    pub presence: PresenceFilterConfig,

    /// Presence + LFU filter configuration.
    ///
    /// Used when "presence_lfu" is in the policies list.
    #[serde(default)]
    #[validate(nested)]
    pub presence_lfu: PresenceLfuFilterConfig,
}

/// Top-level offload configuration.
///
/// Groups policy configurations for each tier transition.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct OffloadConfig {
    /// G1 (GPU) → G2 (Host) offload policies.
    #[serde(default)]
    #[validate(nested)]
    pub g1_to_g2: TierOffloadConfig,

    /// G2 (Host) → G3 (Disk) offload policies.
    #[serde(default)]
    #[validate(nested)]
    pub g2_to_g3: TierOffloadConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OffloadConfig::default();
        // Empty policies - engine applies tier-specific defaults
        assert!(config.g1_to_g2.policies.is_empty());
        assert!(config.g2_to_g3.policies.is_empty());
        assert_eq!(config.g2_to_g3.presence_lfu.min_lfu_count, 8);
    }

    #[test]
    fn test_policy_type_serde() {
        let json = r#"["pass_all", "presence", "presence_lfu"]"#;
        let policies: Vec<PolicyType> = serde_json::from_str(json).unwrap();
        assert_eq!(policies.len(), 3);
        assert_eq!(policies[0], PolicyType::PassAll);
        assert_eq!(policies[1], PolicyType::Presence);
        assert_eq!(policies[2], PolicyType::PresenceLfu);

        // Roundtrip (serde_json doesn't add spaces after commas)
        let serialized = serde_json::to_string(&policies).unwrap();
        let roundtrip: Vec<PolicyType> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(policies, roundtrip);
    }

    #[test]
    fn test_tier_config_serde() {
        let json = r#"{
            "policies": ["presence_lfu"],
            "presence_lfu": { "min_lfu_count": 16 }
        }"#;

        let config: TierOffloadConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.policies.len(), 1);
        assert_eq!(config.policies[0], PolicyType::PresenceLfu);
        assert_eq!(config.presence_lfu.min_lfu_count, 16);
    }

    #[test]
    fn test_offload_config_serde() {
        let json = r#"{
            "g1_to_g2": {
                "policies": ["presence"]
            },
            "g2_to_g3": {
                "policies": ["presence_lfu"],
                "presence_lfu": { "min_lfu_count": 4 }
            }
        }"#;

        let config: OffloadConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.g1_to_g2.policies, vec![PolicyType::Presence]);
        assert_eq!(config.g2_to_g3.policies, vec![PolicyType::PresenceLfu]);
        assert_eq!(config.g2_to_g3.presence_lfu.min_lfu_count, 4);
    }

    #[test]
    fn test_default_lfu_threshold() {
        let json = r#"{"policies": ["presence_lfu"]}"#;
        let config: TierOffloadConfig = serde_json::from_str(json).unwrap();
        // Should use default of 8
        assert_eq!(config.presence_lfu.min_lfu_count, 8);
    }

    #[test]
    fn test_validation() {
        let config = OffloadConfig::default();
        assert!(config.validate().is_ok());

        let config_with_lfu = OffloadConfig {
            g2_to_g3: TierOffloadConfig {
                policies: vec![PolicyType::PresenceLfu],
                presence_lfu: PresenceLfuFilterConfig { min_lfu_count: 1 },
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(config_with_lfu.validate().is_ok());
    }
}
