// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote-search configuration.
//!
//! When `enabled`, a leader consults the hub's KV indexer to locate candidate
//! instances holding a request's locally-uncached blocks (a "remote search"),
//! then pulls them over the transfer control plane. Remote search is only
//! usable when the hub is enabled **and** offers the `indexer` feature; the
//! connector validates that at the hub handshake and fails startup with an
//! invalid-configuration error if remote search is enabled but unavailable.
//!
//! The hub URL is **not** here — it comes from
//! [`KvbmConfig::hub`](crate::KvbmConfig::hub). This block only carries the
//! remote-search toggle + tuning.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Remote-search configuration.
///
/// # JSON example
/// ```json
/// {
///   "enabled": true,
///   "min_remote_tokens": 256
/// }
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct RemoteSearch {
    /// Master toggle. `false` (default) leaves the leader's match path
    /// untouched; the discovery seam is not injected.
    #[serde(default)]
    pub enabled: bool,

    /// Minimum number of locally-uncached ("remote") tokens before a remote
    /// indexer search is worthwhile, expressed in tokens.
    ///
    /// `None` (omitted) → **any remote match**: a search is issued whenever at
    /// least one full remote block remains. `Some(n)` → require at least `n`
    /// tokens' worth of remote data, i.e. `⌈n / block_size⌉` full, complete
    /// blocks. See [`min_remote_blocks`](Self::min_remote_blocks).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_remote_tokens: Option<usize>,
}

impl RemoteSearch {
    /// Block-count threshold derived from [`min_remote_tokens`](Self::min_remote_tokens).
    ///
    /// A remote search is issued when the number of remaining locally-uncached
    /// full blocks is **at least** this value. `None` → `1` (any remote
    /// match); `Some(n)` → `⌈n / block_size⌉`, clamped to a minimum of 1 (a
    /// sub-block round-trip never pays off).
    pub fn min_remote_blocks(&self, block_size: usize) -> usize {
        debug_assert!(block_size > 0, "block_size must be non-zero");
        match self.min_remote_tokens {
            None => 1,
            Some(n) => n.div_ceil(block_size).max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_empty_defaults_disabled() {
        let cfg: RemoteSearch = serde_json::from_str("{}").unwrap();
        assert!(!cfg.enabled);
        assert_eq!(cfg.min_remote_tokens, None);
    }

    #[test]
    fn test_deserialize_explicit() {
        let cfg: RemoteSearch =
            serde_json::from_str(r#"{"enabled": true, "min_remote_tokens": 256}"#).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.min_remote_tokens, Some(256));
    }

    #[test]
    fn test_enabled_without_threshold() {
        let cfg: RemoteSearch = serde_json::from_str(r#"{"enabled": true}"#).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.min_remote_tokens, None);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let cfg = RemoteSearch {
            enabled: true,
            min_remote_tokens: Some(128),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains(r#""enabled":true"#));
        assert!(json.contains(r#""min_remote_tokens":128"#));
        let roundtrip: RemoteSearch = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.min_remote_tokens, cfg.min_remote_tokens);
        assert_eq!(roundtrip.enabled, cfg.enabled);
    }

    #[test]
    fn test_validate_ok() {
        let cfg = RemoteSearch::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_min_remote_blocks_none_is_any_match() {
        let bs = 16;
        // None → any remote match: threshold is one full block.
        assert_eq!(
            RemoteSearch {
                enabled: true,
                min_remote_tokens: None
            }
            .min_remote_blocks(bs),
            1
        );
    }

    #[test]
    fn test_min_remote_blocks_rounds_up_and_clamps() {
        let bs = 16;
        let rs = |n: usize| RemoteSearch {
            enabled: true,
            min_remote_tokens: Some(n),
        };
        // 0 tokens clamps to a 1-block minimum.
        assert_eq!(rs(0).min_remote_blocks(bs), 1);
        // 1 token rounds up to a full block.
        assert_eq!(rs(1).min_remote_blocks(bs), 1);
        // Exactly one block.
        assert_eq!(rs(16).min_remote_blocks(bs), 1);
        // One token into the second block rounds up to 2.
        assert_eq!(rs(17).min_remote_blocks(bs), 2);
        // Several full blocks.
        assert_eq!(rs(256).min_remote_blocks(bs), 16);
    }
}
