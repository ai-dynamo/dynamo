// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Onboard configuration for KV cache loading strategies.
//!
//! This module defines the configuration for how external KV cache blocks
//! are loaded (onboarded) from G2 (host memory) to G1 (GPU memory).

use serde::{Deserialize, Serialize};

/// Configuration for KV cache onboarding strategy.
///
/// Onboarding is the process of loading external KV cache blocks from
/// G2 (host memory) into G1 (GPU memory) for use during inference.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OnboardConfig {
    /// The onboarding mode to use.
    ///
    /// - `inter`: Async out-of-band loading via Nova messages (default)
    /// - `intra`: Synchronous layer-wise loading during forward pass
    #[serde(default)]
    pub mode: OnboardMode,
}

/// Onboarding mode for loading external KV cache blocks.
///
/// This determines when and how G2→G1 transfers occur during inference.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OnboardMode {
    /// Inter-pass onboarding (default).
    ///
    /// Blocks are loaded asynchronously between scheduler passes via Nova
    /// active messages to workers. The `get_num_new_matched_tokens` returns
    /// `(Some(n), true)` to indicate async loading is in progress.
    ///
    /// Pros: Overlaps transfer with computation
    /// Cons: Adds latency before first token if transfer not complete
    #[default]
    Inter,

    /// Intra-pass onboarding.
    ///
    /// Blocks are loaded synchronously during the forward pass, layer by layer.
    /// The `get_num_new_matched_tokens` returns `(Some(n), false)` and the
    /// G2/G1 block pairs are passed to workers via `KvConnectorMetadata`.
    ///
    /// Pros: Guaranteed data availability before each layer
    /// Cons: Serializes transfer with computation per layer
    Intra,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_mode_is_inter() {
        let config = OnboardConfig::default();
        assert_eq!(config.mode, OnboardMode::Inter);
    }

    #[test]
    fn test_mode_serde_roundtrip() {
        // Test inter mode
        let json = r#"{"mode": "inter"}"#;
        let config: OnboardConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, OnboardMode::Inter);

        // Test intra mode
        let json = r#"{"mode": "intra"}"#;
        let config: OnboardConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, OnboardMode::Intra);
    }

    #[test]
    fn test_empty_json_uses_default() {
        let json = r#"{}"#;
        let config: OnboardConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, OnboardMode::Inter);
    }
}
