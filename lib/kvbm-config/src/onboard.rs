// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Onboard configuration for KV cache loading strategies.
//!
//! This module defines the configuration for how external KV cache blocks
//! are loaded (onboarded) from G2 (host memory) to G1 (GPU memory).

use serde::{Deserialize, Serialize};

fn default_stage_chunk_size() -> usize {
    16
}

/// Configuration for KV cache onboarding strategy.
///
/// Onboarding is the process of loading external KV cache blocks from
/// G2 (host memory) into G1 (GPU memory) for use during inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnboardConfig {
    /// The onboarding mode to use.
    ///
    /// - `inter`: Async out-of-band loading via Velo messages (default)
    /// - `intra`: Synchronous layer-wise loading during forward pass
    #[serde(default)]
    pub mode: OnboardMode,

    /// Number of blocks per G3->G2 staging transfer request.
    ///
    /// This is a fixed-size chunk used when staging from disk to host to bound
    /// transfer request size and background notification pressure.
    #[serde(default = "default_stage_chunk_size")]
    pub stage_chunk_size: usize,
}

impl Default for OnboardConfig {
    fn default() -> Self {
        Self {
            mode: OnboardMode::default(),
            stage_chunk_size: default_stage_chunk_size(),
        }
    }
}

/// Onboarding mode for loading external KV cache blocks.
///
/// This determines when and how G2→G1 transfers occur during inference.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OnboardMode {
    /// Inter-pass onboarding (default).
    ///
    /// Blocks are loaded asynchronously between scheduler passes via Velo
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
        assert_eq!(config.stage_chunk_size, 16);
    }

    #[test]
    fn test_mode_serde_roundtrip() {
        // Test inter mode
        let json = r#"{"mode": "inter"}"#;
        let config: OnboardConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, OnboardMode::Inter);
        assert_eq!(config.stage_chunk_size, 16);

        // Test intra mode
        let json = r#"{"mode": "intra"}"#;
        let config: OnboardConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, OnboardMode::Intra);
        assert_eq!(config.stage_chunk_size, 16);
    }

    #[test]
    fn test_stage_chunk_size_serde_roundtrip() {
        let json = r#"{"mode": "inter", "stage_chunk_size": 32}"#;
        let config: OnboardConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, OnboardMode::Inter);
        assert_eq!(config.stage_chunk_size, 32);
    }

    #[test]
    fn test_empty_json_uses_default() {
        let json = r#"{}"#;
        let config: OnboardConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.mode, OnboardMode::Inter);
        assert_eq!(config.stage_chunk_size, 16);
    }
}
