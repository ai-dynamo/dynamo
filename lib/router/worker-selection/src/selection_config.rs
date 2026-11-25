// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router configuration the worker-selection stage reads: scoring weights, sampling
//! temperature, and the shared-cache shape. `dynamo-kv-router` converts its flat
//! `KvRouterConfig` into this once and hands it to policy factories.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Type of external shared KV cache to query during routing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SharedCacheType {
    /// No shared cache (default).
    #[default]
    None,
    /// HiCache L3 shared cache — queries sglang workers via the request plane.
    Hicache,
}

impl fmt::Display for SharedCacheType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Hicache => f.write_str("hicache"),
        }
    }
}

impl FromStr for SharedCacheType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "hicache" => Ok(Self::Hicache),
            _ => Err(format!(
                "unknown shared_cache_type: {s:?}, expected 'none' or 'hicache'"
            )),
        }
    }
}

pub fn validate_overlap_score_credit(value: f64) -> Result<(), String> {
    if value.is_finite() && value >= 0.0 {
        return Ok(());
    }
    Err("overlap_score_credit must be a finite, non-negative number".to_string())
}

pub fn validate_finite_non_negative(field: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value >= 0.0 {
        return Ok(());
    }
    Err(format!("{field} must be finite and non-negative"))
}

pub fn validate_range(field: &str, value: f64, min: f64, max: f64) -> Result<(), String> {
    if value >= min && value <= max {
        return Ok(());
    }
    Err(format!("{field} must be between {min} and {max}"))
}

pub const fn default_overlap_score_credit_decay() -> f64 {
    0.0
}

pub const fn default_prefill_load_scale() -> f64 {
    1.0
}

pub const fn default_decode_active_request_weight() -> f64 {
    0.0
}

pub const fn default_host_cache_hit_weight() -> f64 {
    0.75
}

pub const fn default_disk_cache_hit_weight() -> f64 {
    0.25
}

/// Scoring and sampling parameters every worker-selection policy may read. Field names
/// match the flat router configuration they come from. A plugin reads each value through
/// the accessor of the same name; a host sets one through
/// [`SelectionConfigHostExt`](crate::host::SelectionConfigHostExt).
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionConfig {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    decode_active_request_weight: f64,
    host_cache_hit_weight: f64,
    disk_cache_hit_weight: f64,
    router_temperature: f64,
    shared_cache_multiplier: Option<f64>,
    shared_cache_type: SharedCacheType,
    conditional_disagg_enabled: bool,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            overlap_score_credit: 1.0,
            overlap_score_credit_decay: default_overlap_score_credit_decay(),
            prefill_load_scale: default_prefill_load_scale(),
            decode_active_request_weight: default_decode_active_request_weight(),
            host_cache_hit_weight: default_host_cache_hit_weight(),
            disk_cache_hit_weight: default_disk_cache_hit_weight(),
            router_temperature: 0.0,
            shared_cache_multiplier: None,
            shared_cache_type: SharedCacheType::default(),
            conditional_disagg_enabled: false,
        }
    }
}

impl SelectionConfig {
    /// Credit per cached device block in the default cost function.
    #[inline]
    pub fn overlap_score_credit(&self) -> f64 {
        self.overlap_score_credit
    }

    /// Decay of that credit with the worker's normalized prefill load.
    #[inline]
    pub fn overlap_score_credit_decay(&self) -> f64 {
        self.overlap_score_credit_decay
    }

    /// Weight of projected prefill load.
    #[inline]
    pub fn prefill_load_scale(&self) -> f64 {
        self.prefill_load_scale
    }

    /// Weight of active decode requests.
    #[inline]
    pub fn decode_active_request_weight(&self) -> f64 {
        self.decode_active_request_weight
    }

    /// Credit per host-pinned cached block, relative to a device block.
    #[inline]
    pub fn host_cache_hit_weight(&self) -> f64 {
        self.host_cache_hit_weight
    }

    /// Credit per disk cached block, relative to a device block.
    #[inline]
    pub fn disk_cache_hit_weight(&self) -> f64 {
        self.disk_cache_hit_weight
    }

    /// Softmax temperature of the picker; zero picks the best cost.
    #[inline]
    pub fn router_temperature(&self) -> f64 {
        self.router_temperature
    }

    /// Credit for blocks the shared cache holds beyond the worker's device prefix; `None`
    /// derives it from `shared_cache_type`.
    #[inline]
    pub fn shared_cache_multiplier(&self) -> Option<f64> {
        self.shared_cache_multiplier
    }

    /// Which external shared cache answers block queries.
    #[inline]
    pub fn shared_cache_type(&self) -> SharedCacheType {
        self.shared_cache_type
    }

    /// Whether conditional disaggregation may leave a positive overlap credit on decode.
    #[inline]
    pub fn conditional_disagg_enabled(&self) -> bool {
        self.conditional_disagg_enabled
    }

    /// Rejects non-finite or negative weights and an out-of-range shared-cache multiplier,
    /// with the same messages the flat configuration reports.
    pub fn validate(&self) -> Result<(), String> {
        validate_overlap_score_credit(self.overlap_score_credit)?;
        validate_finite_non_negative(
            "overlap_score_credit_decay",
            self.overlap_score_credit_decay,
        )?;
        validate_finite_non_negative("prefill_load_scale", self.prefill_load_scale)?;
        validate_finite_non_negative(
            "decode_active_request_weight",
            self.decode_active_request_weight,
        )?;
        validate_finite_non_negative("host_cache_hit_weight", self.host_cache_hit_weight)?;
        validate_finite_non_negative("disk_cache_hit_weight", self.disk_cache_hit_weight)?;
        validate_finite_non_negative("router_temperature", self.router_temperature)?;
        if let Some(value) = self.shared_cache_multiplier {
            validate_range("shared_cache_multiplier", value, 0.0, 1.0)?;
        }
        Ok(())
    }
}

impl crate::host::SelectionConfigHostExt for SelectionConfig {
    #[inline]
    fn with_overlap_score_credit(mut self, overlap_score_credit: f64) -> Self {
        self.overlap_score_credit = overlap_score_credit;
        self
    }

    #[inline]
    fn with_overlap_score_credit_decay(mut self, overlap_score_credit_decay: f64) -> Self {
        self.overlap_score_credit_decay = overlap_score_credit_decay;
        self
    }

    #[inline]
    fn with_prefill_load_scale(mut self, prefill_load_scale: f64) -> Self {
        self.prefill_load_scale = prefill_load_scale;
        self
    }

    #[inline]
    fn with_decode_active_request_weight(mut self, decode_active_request_weight: f64) -> Self {
        self.decode_active_request_weight = decode_active_request_weight;
        self
    }

    #[inline]
    fn with_host_cache_hit_weight(mut self, host_cache_hit_weight: f64) -> Self {
        self.host_cache_hit_weight = host_cache_hit_weight;
        self
    }

    #[inline]
    fn with_disk_cache_hit_weight(mut self, disk_cache_hit_weight: f64) -> Self {
        self.disk_cache_hit_weight = disk_cache_hit_weight;
        self
    }

    #[inline]
    fn with_router_temperature(mut self, router_temperature: f64) -> Self {
        self.router_temperature = router_temperature;
        self
    }

    #[inline]
    fn with_shared_cache_multiplier(mut self, shared_cache_multiplier: Option<f64>) -> Self {
        self.shared_cache_multiplier = shared_cache_multiplier;
        self
    }

    #[inline]
    fn with_shared_cache_type(mut self, shared_cache_type: SharedCacheType) -> Self {
        self.shared_cache_type = shared_cache_type;
        self
    }

    #[inline]
    fn with_conditional_disagg_enabled(mut self, conditional_disagg_enabled: bool) -> Self {
        self.conditional_disagg_enabled = conditional_disagg_enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::SelectionConfigHostExt;

    #[test]
    fn each_builder_sets_the_value_its_accessor_returns() {
        let config = SelectionConfig::default()
            .with_overlap_score_credit(2.0)
            .with_overlap_score_credit_decay(0.5)
            .with_prefill_load_scale(3.0)
            .with_decode_active_request_weight(4.0)
            .with_host_cache_hit_weight(0.6)
            .with_disk_cache_hit_weight(0.1)
            .with_router_temperature(0.7)
            .with_shared_cache_multiplier(Some(0.4))
            .with_shared_cache_type(SharedCacheType::Hicache)
            .with_conditional_disagg_enabled(true);

        // Names every field: a new one fails to compile here until it has an accessor, a
        // host builder, and a line in `dynamo-kv-router`'s `From<&KvRouterConfig>`.
        assert_eq!(
            config,
            SelectionConfig {
                overlap_score_credit: 2.0,
                overlap_score_credit_decay: 0.5,
                prefill_load_scale: 3.0,
                decode_active_request_weight: 4.0,
                host_cache_hit_weight: 0.6,
                disk_cache_hit_weight: 0.1,
                router_temperature: 0.7,
                shared_cache_multiplier: Some(0.4),
                shared_cache_type: SharedCacheType::Hicache,
                conditional_disagg_enabled: true,
            }
        );
        assert_eq!(config.overlap_score_credit(), 2.0);
        assert_eq!(config.overlap_score_credit_decay(), 0.5);
        assert_eq!(config.prefill_load_scale(), 3.0);
        assert_eq!(config.decode_active_request_weight(), 4.0);
        assert_eq!(config.host_cache_hit_weight(), 0.6);
        assert_eq!(config.disk_cache_hit_weight(), 0.1);
        assert_eq!(config.router_temperature(), 0.7);
        assert_eq!(config.shared_cache_multiplier(), Some(0.4));
        assert_eq!(config.shared_cache_type(), SharedCacheType::Hicache);
        assert!(config.conditional_disagg_enabled());
    }
}
