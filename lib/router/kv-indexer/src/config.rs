// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The indexer's configuration slice: the router fields that decide how the primary and
//! the predict-on-route side index retain entries. `dynamo-kv-router` converts its flat
//! `KvRouterConfig` into this once.

use std::fmt;
use std::str::FromStr;

use serde::Deserialize;

/// Retention policy for a router-local primary approximate indexer.
///
/// This selector is intentionally process-local. Workers do not advertise it in
/// model cards because request lifetime and release ownership live in the router.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApproximateCachePolicyKind {
    /// Expire predicted entries after `router_ttl_secs`.
    #[default]
    Ttl,
    /// Retain predicted entries until per-rank KV capacity requires LRU eviction.
    Lru,
}

impl fmt::Display for ApproximateCachePolicyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ttl => f.write_str("ttl"),
            Self::Lru => f.write_str("lru"),
        }
    }
}

impl FromStr for ApproximateCachePolicyKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ttl" => Ok(Self::Ttl),
            "lru" => Ok(Self::Lru),
            _ => Err(format!(
                "unknown approximate cache policy {value:?}, expected 'ttl' or 'lru'"
            )),
        }
    }
}

/// Router configuration the indexer reads. Field names drop the `router_` prefix of the
/// flat configuration they come from.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexerConfig {
    /// Engine KV events populate the primary; `false` is approximate (predict-on-route) mode.
    pub use_kv_events: bool,
    /// Retention for a primary approximate indexer.
    pub approximate_cache_policy: ApproximateCachePolicyKind,
    /// TTL of the predict-on-route side indexer next to an event-driven primary; `None` disables it.
    pub predicted_ttl_secs: Option<f64>,
    /// TTL of a primary approximate indexer.
    pub ttl_secs: f64,
}

impl Default for IndexerConfig {
    fn default() -> Self {
        Self {
            use_kv_events: true,
            approximate_cache_policy: ApproximateCachePolicyKind::default(),
            predicted_ttl_secs: None,
            ttl_secs: 120.0,
        }
    }
}
