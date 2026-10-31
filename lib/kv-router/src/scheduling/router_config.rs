// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the kernel reads of the router's configuration, resolved once at construction. The
//! flat `KvRouterConfig` with its environment, CLI, and policy-file parsing is the composed
//! router's; it converts into this slice, and the kernel never reads a file or a variable.

use std::fmt;
use std::ops::Deref;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use dynamo_admission::{PolicyClassConfig, PolicyProfile};
use dynamo_worker_selection::SelectionConfig;
use dynamo_worker_selection::conditional_disagg::ConditionalDisaggConfig;

use dynamo_worker_selection::selection_config::{
    default_disk_cache_hit_weight, default_host_cache_hit_weight, validate_finite_non_negative,
    validate_overlap_score_credit, validate_range,
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

fn apply_deprecated_overlap_score_weight_override_option(
    value: f64,
    overlap_score_credit: &mut Option<f64>,
    prefill_load_scale: &mut Option<f64>,
) {
    *prefill_load_scale = Some(value);
    if value == 0.0 {
        *overlap_score_credit = Some(0.0);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouterPrefillLoadModel {
    #[default]
    None,
    // v1.5 frontends only recognize "aic" in worker cards.
    // TODO(v1.8): write "ais" with alias="aic" when v1.5 leaves N-2.
    // TODO(v1.10): remove the wire "aic" alias when v1.7 writers leave N-2,
    // provided v1.8 and later consistently emit "ais".
    #[serde(rename = "aic", alias = "ais")]
    Ais,
}

impl fmt::Display for RouterPrefillLoadModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Ais => f.write_str("ais"),
        }
    }
}

impl FromStr for RouterPrefillLoadModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "ais" | "aic" => Ok(Self::Ais),
            _ => Err(format!(
                "unknown prefill load model: {s:?}, expected 'none' or 'ais'"
            )),
        }
    }
}

impl RouterPrefillLoadModel {
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RouterConfigOverrideSerde {
    overlap_score_credit: Option<f64>,
    prefill_load_scale: Option<f64>,
    overlap_score_weight: Option<f64>,
    router_temperature: Option<f64>,
    assume_kv_reuse: Option<bool>,
    track_prefill_tokens: Option<bool>,
    shared_cache_multiplier: Option<f64>,
}

/// Override configuration for router settings that can be specified per-request
#[derive(Debug, Clone, Default, Builder, Serialize, Deserialize)]
#[serde(try_from = "RouterConfigOverrideSerde")]
pub struct RouterConfigOverride {
    /// Device-local prefix-overlap credit multiplier applied to the prefill
    /// load before sampling. Values must be finite and non-negative. Values above
    /// 1.0 give device overlap extra credit. Set to 0.0 to ignore prefix matching.
    #[builder(default)]
    pub overlap_score_credit: Option<f64>,

    /// Scale applied to the adjusted prefill load after device/lower-tier
    /// cache-hit credits have been subtracted.
    #[builder(default)]
    pub prefill_load_scale: Option<f64>,

    #[builder(default)]
    pub router_temperature: Option<f64>,

    #[builder(default)]
    pub assume_kv_reuse: Option<bool>,

    #[builder(default)]
    pub track_prefill_tokens: Option<bool>,

    /// Per-request override of `shared_cache_multiplier`.
    #[builder(default)]
    pub shared_cache_multiplier: Option<f64>,
}

impl RouterConfigOverride {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(value) = self.overlap_score_credit {
            validate_overlap_score_credit(value)?;
        }
        if let Some(value) = self.prefill_load_scale {
            validate_finite_non_negative("prefill_load_scale", value)?;
        }
        if let Some(value) = self.router_temperature {
            validate_finite_non_negative("router_temperature", value)?;
        }
        if let Some(value) = self.shared_cache_multiplier {
            validate_range("shared_cache_multiplier", value, 0.0, 1.0)?;
        }
        Ok(())
    }
}

impl TryFrom<RouterConfigOverrideSerde> for RouterConfigOverride {
    type Error = String;

    fn try_from(compat: RouterConfigOverrideSerde) -> Result<Self, Self::Error> {
        let mut overlap_score_credit = compat.overlap_score_credit;
        let mut prefill_load_scale = compat.prefill_load_scale;

        if let Some(overlap_score_weight) = compat.overlap_score_weight {
            apply_deprecated_overlap_score_weight_override_option(
                overlap_score_weight,
                &mut overlap_score_credit,
                &mut prefill_load_scale,
            );
        }

        let config = Self {
            overlap_score_credit,
            prefill_load_scale,
            router_temperature: compat.router_temperature,
            assume_kv_reuse: compat.assume_kv_reuse,
            track_prefill_tokens: compat.track_prefill_tokens,
            shared_cache_multiplier: compat.shared_cache_multiplier,
        };
        config.validate()?;
        Ok(config)
    }
}
