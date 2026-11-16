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

pub const OVERLAP_SCORE_CREDIT_RANGE_ERROR: &str =
    "overlap_score_credit must be a finite, non-negative number";

pub fn overlap_score_credit_error_message(value: f64) -> Option<&'static str> {
    if value.is_finite() && value >= 0.0 {
        None
    } else {
        Some(OVERLAP_SCORE_CREDIT_RANGE_ERROR)
    }
}

pub fn validate_overlap_score_credit(value: f64) -> Result<(), String> {
    let Some(message) = overlap_score_credit_error_message(value) else {
        return Ok(());
    };
    Err(message.to_string())
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
/// match the flat router configuration they come from.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionConfig {
    /// Credit per cached device block in the default cost function.
    pub overlap_score_credit: f64,
    /// Decay of that credit with the worker's normalized prefill load.
    pub overlap_score_credit_decay: f64,
    /// Weight of projected prefill load.
    pub prefill_load_scale: f64,
    /// Weight of active decode requests.
    pub decode_active_request_weight: f64,
    /// Credit per host-pinned cached block, relative to a device block.
    pub host_cache_hit_weight: f64,
    /// Credit per disk cached block, relative to a device block.
    pub disk_cache_hit_weight: f64,
    /// Softmax temperature of the picker; zero picks the best cost.
    pub router_temperature: f64,
    /// Credit for blocks the shared cache holds beyond the worker's device prefix; `None`
    /// derives it from `shared_cache_type`.
    pub shared_cache_multiplier: Option<f64>,
    /// Which external shared cache answers block queries.
    pub shared_cache_type: SharedCacheType,
    /// Whether conditional disaggregation may leave a positive overlap credit on decode.
    pub conditional_disagg_enabled: bool,
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
