// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer preferences for resolving redundant strategy choices.
//!
//! Some source/destination combinations can use multiple transfer strategies.
//! For example:
//! - System ↔ Pinned: memcpy or NIXL
//! - Pinned ↔ Device: CUDA or NIXL
//!
//! This module provides preferences to control which strategy to prefer.

use serde::{Deserialize, Serialize};

/// Policy for choosing between native transports (memcpy/CUDA) and NIXL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NativeVsNixlPolicy {
    /// Always prefer native transports (memcpy/CUDA) when available
    PreferNative,

    /// Always prefer NIXL when available
    PreferNixl,

    /// Use native for local-to-local, NIXL for remote/disk
    #[default]
    Automatic,
}

/// Transfer preferences for strategy selection.
///
/// These preferences allow fine-grained control over transfer strategy selection
/// when multiple valid strategies exist for a source/destination pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferPreferences {
    /// Policy for native vs NIXL transport selection
    pub native_vs_nixl: NativeVsNixlPolicy,

    /// Whether to prefer async CUDA operations over blocking ones
    pub prefer_async_cuda: bool,
}

impl Default for TransferPreferences {
    fn default() -> Self {
        Self {
            native_vs_nixl: NativeVsNixlPolicy::default(),
            prefer_async_cuda: true,
        }
    }
}

impl TransferPreferences {
    /// Create preferences with all defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create preferences that always prefer native transports.
    pub fn prefer_native() -> Self {
        Self {
            native_vs_nixl: NativeVsNixlPolicy::PreferNative,
            prefer_async_cuda: true,
        }
    }

    /// Create preferences that always prefer NIXL.
    pub fn prefer_nixl() -> Self {
        Self {
            native_vs_nixl: NativeVsNixlPolicy::PreferNixl,
            prefer_async_cuda: true,
        }
    }

    /// Set the native vs NIXL policy.
    pub fn with_native_vs_nixl(mut self, policy: NativeVsNixlPolicy) -> Self {
        self.native_vs_nixl = policy;
        self
    }

    /// Set whether to prefer async CUDA operations.
    pub fn with_async_cuda(mut self, prefer_async: bool) -> Self {
        self.prefer_async_cuda = prefer_async;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_preferences() {
        let prefs = TransferPreferences::default();
        assert_eq!(prefs.native_vs_nixl, NativeVsNixlPolicy::Automatic);
        assert!(prefs.prefer_async_cuda);
    }

    #[test]
    fn test_prefer_native() {
        let prefs = TransferPreferences::prefer_native();
        assert_eq!(prefs.native_vs_nixl, NativeVsNixlPolicy::PreferNative);
        assert!(prefs.prefer_async_cuda);
    }

    #[test]
    fn test_prefer_nixl() {
        let prefs = TransferPreferences::prefer_nixl();
        assert_eq!(prefs.native_vs_nixl, NativeVsNixlPolicy::PreferNixl);
        assert!(prefs.prefer_async_cuda);
    }

    #[test]
    fn test_builder_pattern() {
        let prefs = TransferPreferences::new()
            .with_native_vs_nixl(NativeVsNixlPolicy::PreferNixl)
            .with_async_cuda(false);

        assert_eq!(prefs.native_vs_nixl, NativeVsNixlPolicy::PreferNixl);
        assert!(!prefs.prefer_async_cuda);
    }
}
