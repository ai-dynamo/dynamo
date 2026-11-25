// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router configuration a model deployment card carries: the routing mode and its
//! KV router settings, the worker load thresholds, and the session affinity policy.

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_session_affinity::host::SessionAffinityMode;
use dynamo_slot_tracker::host::{BusyThresholds, BusyThresholdsHostExt};
use serde::{Deserialize, Serialize};

/// Configuration for worker load thresholds used in overload detection.
///
/// All thresholds are opt-in. An unset (`None`) field means the corresponding
/// check is skipped entirely — it never contributes to a worker being marked
/// overloaded. If all three are `None`, overload-based rejection is fully disabled.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LoadThresholdConfig {
    /// KV cache block utilization threshold (0.0-1.0).
    /// Worker is overloaded when `active_decode_blocks / total_blocks > threshold`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_decode_blocks_threshold: Option<f64>,

    /// Absolute prefill token count threshold.
    /// Worker is overloaded when `active_prefill_tokens > threshold`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_prefill_tokens_threshold: Option<u64>,

    /// Fraction of max_num_batched_tokens.
    /// Worker is overloaded when `active_prefill_tokens > frac * max_num_batched_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_prefill_tokens_threshold_frac: Option<f64>,
}

impl LoadThresholdConfig {
    /// Returns true if any threshold is configured.
    pub fn is_configured(&self) -> bool {
        self.active_decode_blocks_threshold.is_some()
            || self.active_prefill_tokens_threshold.is_some()
            || self.active_prefill_tokens_threshold_frac.is_some()
    }

    /// Validate threshold values shared by startup and dynamic configuration.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(threshold) = self.active_decode_blocks_threshold
            && (!threshold.is_finite() || !(0.0..=1.0).contains(&threshold))
        {
            return Err(format!(
                "active_decode_blocks_threshold must be between 0.0 and 1.0, got {threshold}"
            ));
        }

        if let Some(threshold) = self.active_prefill_tokens_threshold_frac
            && (!threshold.is_finite() || threshold < 0.0)
        {
            return Err(format!(
                "active_prefill_tokens_threshold_frac must be a finite value greater than or equal to 0.0, got {threshold}"
            ));
        }

        Ok(())
    }
}

impl From<LoadThresholdConfig> for BusyThresholds {
    /// The queue's vocabulary for the `/busy_threshold` config: token counts narrow to
    /// `usize` saturating, fractions pass through.
    fn from(config: LoadThresholdConfig) -> Self {
        BusyThresholds::default()
            .with_prefill_busy_threshold(
                config
                    .active_prefill_tokens_threshold
                    .map(|tokens| usize::try_from(tokens).unwrap_or(usize::MAX)),
            )
            .with_prefill_busy_threshold_frac(config.active_prefill_tokens_threshold_frac)
            .with_decode_busy_threshold_frac(config.active_decode_blocks_threshold)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterConfig {
    pub router_mode: RouterMode,
    pub kv_router_config: KvRouterConfig,
    /// Load threshold configuration for overload detection
    pub load_threshold_config: LoadThresholdConfig,
    /// Deprecated compatibility field. Routing and readiness ignore this value.
    #[serde(default)]
    pub enforce_disagg: bool,
    #[serde(default)]
    pub session_affinity_ttl_secs: Option<u64>,
    #[serde(default)]
    pub session_affinity_mode: SessionAffinityMode,
}

impl RouterConfig {
    pub fn new(router_mode: RouterMode, kv_router_config: KvRouterConfig) -> Self {
        Self {
            router_mode,
            kv_router_config,
            load_threshold_config: LoadThresholdConfig::default(),
            enforce_disagg: false,
            session_affinity_ttl_secs: None,
            session_affinity_mode: SessionAffinityMode::Hard,
        }
    }

    pub fn with_load_threshold_config(mut self, config: LoadThresholdConfig) -> Self {
        self.load_threshold_config = config;
        self
    }

    #[deprecated(
        note = "enforce_disagg is ignored; topology and readiness come from registered worker types"
    )]
    pub fn with_enforce_disagg(mut self, enforce_disagg: bool) -> Self {
        self.enforce_disagg = enforce_disagg;
        self
    }

    pub fn with_session_affinity_ttl_secs(mut self, ttl_secs: u64) -> Self {
        self.session_affinity_ttl_secs = Some(ttl_secs);
        self
    }
}
