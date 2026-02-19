// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use rand::Rng;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::kv_router::protocols::{compute_block_hash_for_seq, compute_seq_hash_for_block};

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
}

/// KV Router configuration parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_kv_router_config"))]
pub struct KvRouterConfig {
    #[validate(range(min = 0.0))]
    pub overlap_score_weight: f64,

    #[validate(range(min = 0.0))]
    pub router_temperature: f64,

    pub use_kv_events: bool,

    /// Enable durable KV events using NATS JetStream instead of the default event plane.
    /// When false (default), the router uses the event-plane subscriber and requires
    /// workers to have local_indexer enabled for gap recovery.
    /// When true, uses JetStream for durability and multi-replica consistency.
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
    /// If None (default), queueing is disabled and all requests go directly to ready.
    /// Must be > 0.
    #[validate(range(min = 0.0))]
    pub router_queue_threshold: Option<f64>,

    /// Number of event processing threads for the KV indexer.
    /// When > 1, uses ConcurrentRadixTree with a thread pool instead of the
    /// single-threaded RadixTree. Default: 1.
    #[validate(range(min = 1))]
    pub router_event_threads: u32,
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
            router_snapshot_threshold: Some(1000000),
            router_reset_states: false,
            router_ttl_secs: 120.0,
            router_max_tree_size: 2usize.pow(20), // 2^20 = 1048576, matches PruneConfig::default()
            router_prune_target_ratio: 0.8,
            router_queue_threshold: None,
            router_event_threads: 1,
        }
    }
}

fn validate_kv_router_config(config: &KvRouterConfig) -> Result<(), ValidationError> {
    if config.durable_kv_events && !config.use_kv_events {
        return Err(ValidationError::new(
            "durable_kv_events requires use_kv_events=true",
        ));
    }
    if !config.use_kv_events && config.router_event_threads > 1 {
        return Err(ValidationError::new(
            "router_event_threads > 1 requires use_kv_events=true",
        ));
    }
    if config.router_track_output_blocks && !config.router_track_active_blocks {
        return Err(ValidationError::new(
            "router_track_output_blocks requires router_track_active_blocks=true",
        ));
    }
    Ok(())
}

impl KvRouterConfig {
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
    ) -> Option<Vec<u64>> {
        if !self.router_track_active_blocks {
            return None;
        }

        let num_blocks = tokens.len() / block_size as usize;
        if num_blocks == 0 {
            return Some(Vec::new());
        }

        // Use override if provided, otherwise use default config
        let assume_kv_reuse = config_override
            .and_then(|cfg| cfg.assume_kv_reuse)
            .unwrap_or(self.router_assume_kv_reuse);

        if assume_kv_reuse {
            // Compute actual block hashes and sequence hashes
            let block_hashes = compute_block_hash_for_seq(tokens, block_size, None);
            Some(compute_seq_hash_for_block(&block_hashes))
        } else {
            // Generate random hashes (no KV reuse assumed)
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
