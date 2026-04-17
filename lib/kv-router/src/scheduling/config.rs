// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env::{self, VarError};
use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use derive_builder::Builder;
use rand::Rng;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::protocols::{
    BlockHashOptions, LocalBlockHash, compute_block_hash_for_seq, compute_seq_hash_for_block,
};

const fn default_track_prefill_tokens() -> bool {
    true
}

pub const DYN_ROUTER_MIN_INITIAL_WORKERS: &str = "DYN_ROUTER_MIN_INITIAL_WORKERS";
pub const DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS: usize = 5000;

pub fn min_initial_workers_from_env() -> anyhow::Result<usize> {
    match env::var(DYN_ROUTER_MIN_INITIAL_WORKERS) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            anyhow::anyhow!(
                "{DYN_ROUTER_MIN_INITIAL_WORKERS} must be a non-negative integer, got {value:?}: {error}"
            )
        }),
        Err(VarError::NotPresent) => Ok(0),
        Err(VarError::NotUnicode(_)) => {
            anyhow::bail!("{DYN_ROUTER_MIN_INITIAL_WORKERS} must be valid unicode")
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterQueuePolicy {
    #[default]
    Fcfs,
    Lcfs,
    Wspt,
}

impl fmt::Display for RouterQueuePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fcfs => f.write_str("fcfs"),
            Self::Lcfs => f.write_str("lcfs"),
            Self::Wspt => f.write_str("wspt"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterPrefillLoadModel {
    #[default]
    None,
    Aic,
}

impl fmt::Display for RouterPrefillLoadModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Aic => f.write_str("aic"),
        }
    }
}

impl FromStr for RouterPrefillLoadModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "aic" => Ok(Self::Aic),
            _ => Err(format!(
                "unknown prefill load model: {s:?}, expected 'none' or 'aic'"
            )),
        }
    }
}

impl RouterPrefillLoadModel {
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl FromStr for RouterQueuePolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fcfs" => Ok(Self::Fcfs),
            "lcfs" => Ok(Self::Lcfs),
            "wspt" => Ok(Self::Wspt),
            _ => Err(format!(
                "unknown queue policy: {s:?}, expected 'fcfs', 'lcfs', or 'wspt'"
            )),
        }
    }
}

/// Override configuration for router settings that can be specified per-request
#[derive(Debug, Clone, Default, Builder, Serialize, Deserialize, Validate)]
pub struct RouterConfigOverride {
    #[builder(default)]
    pub overlap_score_weight: Option<f64>,

    #[builder(default)]
    #[validate(range(min = 0.0))]
    pub router_temperature: Option<f64>,

    #[builder(default)]
    pub assume_kv_reuse: Option<bool>,

    #[builder(default)]
    pub track_prefill_tokens: Option<bool>,
}

/// KV Router configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(default)]
#[validate(schema(function = "validate_kv_router_config"))]
pub struct KvRouterConfig {
    #[validate(range(min = 0.0))]
    pub overlap_score_weight: f64,

    #[validate(range(min = 0.0))]
    pub router_temperature: f64,

    pub use_kv_events: bool,

    /// **Deprecated:** Enable durable KV events using NATS JetStream instead of the default event plane.
    /// This option will be removed in a future release. The event-plane subscriber
    /// (local_indexer mode) is now the recommended path.
    pub durable_kv_events: bool,

    pub router_replica_sync: bool,

    /// Whether to track active blocks in the router (default: true)
    pub router_track_active_blocks: bool,

    /// Whether to track output blocks during generation (default: false)
    /// When enabled, the router adds placeholder blocks as tokens are generated
    /// and applies fractional decay based on progress toward agent_hints.osl.
    pub router_track_output_blocks: bool,

    /// Whether to assume KV cache reuse when tracking active blocks (default: true).
    /// When true, computes actual block hashes for sequence tracking.
    /// When false, generates random hashes (assuming no KV cache reuse).
    pub router_assume_kv_reuse: bool,

    /// Whether to include prompt-side prefill tokens in active load accounting (default: true).
    /// When false, prompt tokens are excluded from active prefill token tracking, queue pressure,
    /// and potential prefill-token load calculations.
    #[serde(default = "default_track_prefill_tokens")]
    pub router_track_prefill_tokens: bool,

    /// Optional model for estimating effective prompt-side prefill load over time.
    pub router_prefill_load_model: RouterPrefillLoadModel,

    /// Threshold for triggering snapshots. If None, no snapshots will be performed.
    #[validate(range(min = 1))]
    pub router_snapshot_threshold: Option<u32>,

    /// Whether to reset the router state on startup (default: false)
    pub router_reset_states: bool,

    /// TTL for blocks in seconds (only used when use_kv_events is false, default: 120.0)
    #[validate(range(min = 0.0))]
    pub router_ttl_secs: f64,

    /// Maximum tree size before pruning (only used when use_kv_events is false, default: 2^20 = 1048576)
    #[validate(range(min = 1))]
    pub router_max_tree_size: usize,

    /// Target size ratio after pruning (only used when use_kv_events is false, default: 0.8)
    #[validate(range(min = 0.0, max = 1.0))]
    pub router_prune_target_ratio: f64,

    /// Queue threshold fraction for prefill token capacity.
    /// When set, requests are queued if all workers exceed this fraction of max_num_batched_tokens.
    /// If None, queueing is disabled and all requests go directly to ready.
    /// Default: 4.0. Must be >= 0. Use 0.0 for maximum queueing sensitivity.
    #[validate(range(min = 0.0))]
    pub router_queue_threshold: Option<f64>,

    /// Number of event processing threads for the KV indexer.
    /// When > 1, uses ConcurrentRadixTree with a thread pool instead of the
    /// single-threaded RadixTree. Default: 4.
    #[validate(range(min = 1))]
    pub router_event_threads: u32,

    pub skip_initial_worker_wait: bool,

    /// Scheduling policy for the router queue.
    /// "fcfs" (default): first-come first-served with priority bumps — optimizes tail TTFT.
    /// "wspt": weighted shortest processing time (Smith's rule) — optimizes average TTFT.
    pub router_queue_policy: RouterQueuePolicy,

    /// Whether to query a remote KV indexer served from the worker component
    /// instead of maintaining a local radix tree for overlap scoring.
    #[serde(default)]
    pub use_remote_indexer: bool,

    /// Whether this router should serve its local indexer from the worker component.
    /// This enables other routers/frontends in the same namespace to query
    /// overlap scores remotely over the request plane by component + endpoint.
    #[serde(default)]
    pub serve_indexer: bool,

    /// Whether the frontend may skip remote prefill and route directly to a decode worker when
    /// the decode worker only needs to compute a bounded number of net-new prompt tokens.
    #[serde(default)]
    pub conditional_prefill_enabled: bool,

    /// Maximum net-new prompt tokens allowed for conditional local prefill on a decode worker.
    pub conditional_prefill_max_new_tokens: usize,
}

impl Default for KvRouterConfig {
    fn default() -> Self {
        Self {
            overlap_score_weight: 1.0,
            router_temperature: 0.0,
            use_kv_events: true,
            durable_kv_events: false, // default to NATS Core (local indexer mode)
            router_replica_sync: false,
            router_track_active_blocks: true,
            router_track_output_blocks: false,
            router_assume_kv_reuse: true,
            router_track_prefill_tokens: default_track_prefill_tokens(),
            router_prefill_load_model: RouterPrefillLoadModel::default(),
            router_snapshot_threshold: Some(1000000),
            router_reset_states: false,
            router_ttl_secs: 120.0,
            router_max_tree_size: 2usize.pow(20), // 2^20 = 1048576, matches PruneConfig::default()
            router_prune_target_ratio: 0.8,
            router_queue_threshold: Some(4.0),
            router_event_threads: 4,
            skip_initial_worker_wait: false,
            router_queue_policy: RouterQueuePolicy::default(),
            use_remote_indexer: false,
            serve_indexer: false,
            conditional_prefill_enabled: false,
            conditional_prefill_max_new_tokens: DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS,
        }
    }
}

fn validate_kv_router_config(config: &KvRouterConfig) -> Result<(), ValidationError> {
    if config.durable_kv_events {
        tracing::warn!(
            "--durable-kv-events is deprecated and will be removed in a future release. \
             The event-plane subscriber (local_indexer mode) is now the recommended path."
        );
    }
    if config.durable_kv_events && !config.use_kv_events {
        return Err(ValidationError::new(
            "durable_kv_events requires use_kv_events=true",
        ));
    }
    if config.router_track_output_blocks && !config.router_track_active_blocks {
        return Err(ValidationError::new(
            "router_track_output_blocks requires router_track_active_blocks=true",
        ));
    }
    if config.router_prefill_load_model.is_enabled() && !config.router_track_prefill_tokens {
        return Err(ValidationError::new(
            "router_prefill_load_model requires router_track_prefill_tokens=true",
        ));
    }
    if config.router_prefill_load_model.is_enabled()
        && !matches!(config.router_queue_policy, RouterQueuePolicy::Fcfs)
    {
        return Err(ValidationError::new(
            "router_prefill_load_model currently requires router_queue_policy='fcfs'",
        ));
    }
    if config.use_remote_indexer && config.serve_indexer {
        return Err(ValidationError::new(
            "use_remote_indexer and serve_indexer are mutually exclusive",
        ));
    }
    if config.serve_indexer && config.overlap_score_weight == 0.0 {
        return Err(ValidationError::new(
            "serve_indexer requires overlap_score_weight > 0",
        ));
    }
    Ok(())
}

impl KvRouterConfig {
    pub fn router_queue_recheck_interval(&self) -> Duration {
        const DEFAULT_RECHECK_INTERVAL: Duration = Duration::from_secs(60);
        const PREFILL_LOAD_RECHECK_INTERVAL: Duration = Duration::from_millis(100);

        if self.router_prefill_load_model.is_enabled() && self.router_queue_threshold.is_some() {
            return PREFILL_LOAD_RECHECK_INTERVAL;
        }

        DEFAULT_RECHECK_INTERVAL
    }

    pub fn assume_kv_reuse(&self, config_override: Option<&RouterConfigOverride>) -> bool {
        config_override
            .and_then(|cfg| cfg.assume_kv_reuse)
            .unwrap_or(self.router_assume_kv_reuse)
    }

    pub fn track_prefill_tokens(&self, config_override: Option<&RouterConfigOverride>) -> bool {
        config_override
            .and_then(|cfg| cfg.track_prefill_tokens)
            .unwrap_or(self.router_track_prefill_tokens)
    }

    /// Compute sequence hashes for active block tracking based on configuration.
    ///
    /// Returns:
    /// - `None` if `router_track_active_blocks` is false
    /// - Random hashes if `router_track_active_blocks` is true but `router_assume_kv_reuse` is false
    /// - Actual sequence hashes if both are true
    pub fn compute_seq_hashes_for_tracking(
        &self,
        tokens: &[u32],
        block_size: u32,
        config_override: Option<&RouterConfigOverride>,
        hash_options: BlockHashOptions<'_>,
        precomputed_block_hashes: Option<&[LocalBlockHash]>,
    ) -> Option<Vec<u64>> {
        if !self.router_track_active_blocks {
            return None;
        }

        let num_blocks = tokens.len() / block_size as usize;
        if num_blocks == 0 {
            return Some(Vec::new());
        }

        let assume_kv_reuse = self.assume_kv_reuse(config_override);

        if assume_kv_reuse {
            let block_hashes = match precomputed_block_hashes {
                Some(block_hashes) => block_hashes,
                None => {
                    let computed = compute_block_hash_for_seq(tokens, block_size, hash_options);
                    return Some(compute_seq_hash_for_block(&computed));
                }
            };
            Some(compute_seq_hash_for_block(block_hashes))
        } else {
            let mut rng = rand::rng();
            Some((0..num_blocks).map(|_| rng.random::<u64>()).collect())
        }
    }

    /// Check if KV event subscription should be started.
    ///
    /// Returns false if:
    /// - KV events are disabled (`use_kv_events=false`)
    /// - Overlap scoring is disabled (`overlap_score_weight=0`)
    ///
    /// When false, the router skips starting the KV event subscription entirely,
    /// avoiding the need to query workers for their local indexer state.
    pub fn should_subscribe_to_kv_events(&self) -> bool {
        self.use_kv_events && self.overlap_score_weight > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{BlockExtraInfo, BlockMmObjectInfo};

    #[test]
    fn compute_seq_hashes_for_tracking_uses_mm_hashes() {
        let cfg = KvRouterConfig::default();
        let tokens = vec![1, 2, 3, 4];
        let mm_infos = vec![
            Some(BlockExtraInfo {
                mm_objects: vec![BlockMmObjectInfo {
                    mm_hash: 42,
                    offsets: vec![],
                }],
            }),
            None,
        ];

        let without_mm = cfg
            .compute_seq_hashes_for_tracking(&tokens, 2, None, BlockHashOptions::default(), None)
            .unwrap();
        let with_mm = cfg
            .compute_seq_hashes_for_tracking(
                &tokens,
                2,
                None,
                BlockHashOptions {
                    block_mm_infos: Some(&mm_infos),
                    ..Default::default()
                },
                None,
            )
            .unwrap();

        assert_ne!(without_mm, with_mm);
    }

    #[test]
    fn compute_seq_hashes_for_tracking_uses_precomputed_block_hashes() {
        let config = KvRouterConfig::default();
        let tokens: Vec<u32> = (0..8).collect();
        let precomputed = vec![LocalBlockHash(11), LocalBlockHash(29)];

        let seq_hashes = config.compute_seq_hashes_for_tracking(
            &tokens,
            4,
            None,
            BlockHashOptions::default(),
            Some(&precomputed),
        );

        assert_eq!(seq_hashes, Some(compute_seq_hash_for_block(&precomputed)));
    }

    #[test]
    fn conditional_prefill_defaults_to_disabled_with_token_cap() {
        let config = KvRouterConfig::default();

        assert!(!config.conditional_prefill_enabled);
        assert_eq!(
            config.conditional_prefill_max_new_tokens,
            DEFAULT_CONDITIONAL_PREFILL_MAX_NEW_TOKENS
        );
    }
}
