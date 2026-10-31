// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the kernel reads of the router's configuration, resolved once at construction. The
//! flat `KvRouterConfig` with its environment, CLI, and policy-file parsing is the composed
//! router's; it converts into this slice, and the kernel never reads a file or a variable.

use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

use dynamo_admission::{PolicyClassConfig, PolicyProfile};
use dynamo_worker_selection::SelectionConfig;
use dynamo_worker_selection::conditional_disagg::ConditionalDisaggConfig;

use super::config::RouterConfigOverride;
use dynamo_worker_selection::selection_config::{
    default_disk_cache_hit_weight, default_host_cache_hit_weight,
};

/// Why a [`RouterConfig`] is not valid.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RouterConfigError {
    #[error("invalid router config: {0}")]
    Validation(String),
}

/// How much a prefix held in a lower cache tier counts toward a rank's effective overlap,
/// relative to a device hit. The knob lives in the worker-selection slice, where policies
/// that score tiers themselves read it; the kernel's overlap analysis reads the same two
/// values through this view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheHitWeights {
    /// Host-pinned memory.
    pub host: f64,
    /// Disk and external tiers.
    pub disk: f64,
}

impl Default for CacheHitWeights {
    fn default() -> Self {
        Self {
            host: default_host_cache_hit_weight(),
            disk: default_disk_cache_hit_weight(),
        }
    }
}

impl From<&SelectionConfig> for CacheHitWeights {
    #[inline]
    fn from(config: &SelectionConfig) -> Self {
        Self {
            host: config.host_cache_hit_weight,
            disk: config.disk_cache_hit_weight,
        }
    }
}

/// Resolves the policy profile a partition schedules under, by model name. The composed
/// router answers from its policy file; a synthetic profile when there is none.
pub type PolicyProfileSource =
    Arc<dyn Fn(Option<&str>) -> Result<PolicyProfile, RouterConfigError> + Send + Sync>;

/// The kernel's configuration.
#[derive(Clone)]
pub struct RouterConfig {
    /// The worker-selection stage's slice, handed to the policy factory.
    pub selection: SelectionConfig,
    /// The conditional-disaggregation slice; the kernel reads its busy thresholds when it
    /// reports a selected worker's load to the prefill router.
    pub conditional_disagg: ConditionalDisaggConfig,
    /// Whether partitions listen for worker KV events (as opposed to approximate tracking).
    pub use_kv_events: bool,
    /// Whether requests count toward prefill load unless the request says otherwise.
    pub track_prefill_tokens: bool,
    /// Whether a request's blocks are assumed reused by the worker it was routed to, unless
    /// the request says otherwise.
    pub assume_kv_reuse: bool,
    /// How often a held queue re-evaluates its workers without a triggering event.
    pub queue_recheck_interval: Duration,
    pub profiles: PolicyProfileSource,
}

impl std::fmt::Debug for RouterConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RouterConfig")
            .field("selection", &self.selection)
            .field("conditional_disagg", &self.conditional_disagg)
            .field("use_kv_events", &self.use_kv_events)
            .field("track_prefill_tokens", &self.track_prefill_tokens)
            .field("assume_kv_reuse", &self.assume_kv_reuse)
            .field("queue_recheck_interval", &self.queue_recheck_interval)
            .finish_non_exhaustive()
    }
}

impl RouterConfig {
    /// Whether to assume KV reuse for a request, honoring its override.
    pub fn assume_kv_reuse(&self, config_override: Option<&RouterConfigOverride>) -> bool {
        config_override
            .and_then(|config| config.assume_kv_reuse)
            .unwrap_or(self.assume_kv_reuse)
    }

    /// Whether any class of `model_name`'s profile holds requests instead of rejecting them.
    pub fn queueing_enabled(&self, model_name: Option<&str>) -> Result<bool, RouterConfigError> {
        Ok((self.profiles)(model_name)?
            .classes()
            .iter()
            .any(PolicyClassConfig::queueing_enabled))
    }

    /// Check every field the kernel reads; the result is the only config a kernel
    /// constructor accepts.
    pub fn validate(self) -> Result<Validated<Self>, RouterConfigError> {
        self.selection
            .validate()
            .map_err(RouterConfigError::Validation)?;
        if self.queue_recheck_interval.is_zero() {
            return Err(RouterConfigError::Validation(
                "queue_recheck_interval must be positive".to_string(),
            ));
        }
        Ok(Validated(self))
    }
}

/// A configuration that passed its own `validate`; constructed nowhere else.
#[derive(Debug, Clone)]
pub struct Validated<T>(T);

impl<T> Deref for Validated<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> Validated<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}
