// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LoRA Allocation Configuration

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::lora::routing::AllocationAlgorithmType;
use dynamo_runtime::config::environment_names::llm;

/// Which predictor to use for smoothing per-LoRA load estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictorType {
    /// Raw bucketed counts (no smoothing).
    None,
    /// Exponential moving average.
    Ema,
}

impl FromStr for PredictorType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "ema" => Ok(Self::Ema),
            other => Err(format!(
                "unknown predictor type: {other:?} (expected \"none\" or \"ema\")"
            )),
        }
    }
}

/// Configuration for the Min-Cost Flow placement solver.
///
/// Read from the `DYN_LORA_MCF_CONFIG` env var as JSON. Omitted fields
/// fall back to defaults via `#[serde(default)]`.
///
/// Example: `{"candidate_m":16,"gamma_load":2000,"beta_keep":500}`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct McfConfig {
    /// Number of HRW top-M candidates per LoRA.
    pub candidate_m: usize,
    /// Preference weight for HRW rank ordering.
    pub alpha_pref: i64,
    /// Penalty weight for loading a new LoRA on a worker.
    pub gamma_load: i64,
    /// Reward weight for keeping a LoRA on its prior worker.
    pub beta_keep: i64,
    /// Cost assigned to the overflow dummy worker.
    pub overflow_cost: i64,
    /// Whether to allow overflow (soft infeasibility) or fail hard.
    pub allow_overflow: bool,
    /// Default churn weight per LoRA (uniform=1; can be per-adapter later).
    pub churn_weight_default: i64,
}

impl Default for McfConfig {
    fn default() -> Self {
        Self {
            candidate_m: 16,
            alpha_pref: 1,
            gamma_load: 1000,
            beta_keep: 250,
            overflow_cost: 1_000_000_000_000,
            allow_overflow: true,
            churn_weight_default: 1,
        }
    }
}

/// Configuration for the LoRA allocation controller.
#[derive(Debug, Clone)]
pub struct LoraAllocationConfig {
    pub enabled: bool,
    pub algorithm: AllocationAlgorithmType,
    /// How often (in seconds) the controller recomputes allocations.
    pub timestep_secs: u64,
    /// Ticks to wait before scaling down a LoRA's replicas.
    pub scale_down_cooldown_ticks: u32,
    /// Multiplier for the load estimator's rate window relative to the controller timestep.
    pub rate_window_multiplier: u64,
    /// Number of counter buckets per second in the BucketedRateCounter.
    pub buckets_per_second: u64,
    pub predictor_type: PredictorType,
    /// EMA smoothing factor (alpha). Range [0.0, 1.0].
    pub ema_alpha: f64,
    /// MCF-specific configuration (only used when algorithm = MinCostFlow).
    pub mcf: McfConfig,
}

/// Minimum rate window (seconds).
pub const MIN_RATE_WINDOW_SECS: u64 = 5;

impl Default for LoraAllocationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: AllocationAlgorithmType::Hrw,
            timestep_secs: 3,
            scale_down_cooldown_ticks: 3,
            rate_window_multiplier: 30,
            buckets_per_second: 1,
            predictor_type: PredictorType::Ema,
            ema_alpha: 0.3,
            mcf: McfConfig::default(),
        }
    }
}

impl LoraAllocationConfig {
    pub fn new(
        enabled: bool,
        algorithm: &str,
        timestep_secs: u64,
        scale_down_cooldown_ticks: u32,
        rate_window_multiplier: u64,
    ) -> Result<Self, String> {
        Ok(Self {
            enabled,
            algorithm: AllocationAlgorithmType::from_str(algorithm)?,
            timestep_secs,
            scale_down_cooldown_ticks,
            rate_window_multiplier,
            ..Default::default()
        })
    }

    /// Compute the effective rate window (seconds) for the load estimator.
    pub fn effective_rate_window_secs(&self) -> u64 {
        (self.timestep_secs * self.rate_window_multiplier).max(MIN_RATE_WINDOW_SECS)
    }

    /// Create config from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let defaults = Self::default();

        let enabled = std::env::var(llm::DYN_LORA_ALLOCATION_ENABLED)
            .map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"))
            .unwrap_or(defaults.enabled);

        let algorithm = std::env::var(llm::DYN_LORA_ALLOCATION_ALGORITHM)
            .ok()
            .and_then(|v| AllocationAlgorithmType::from_str(&v).ok())
            .unwrap_or(defaults.algorithm);

        let timestep_secs = std::env::var(llm::DYN_LORA_ALLOCATION_TIMESTEP_SECS)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.timestep_secs);

        let scale_down_cooldown_ticks =
            std::env::var(llm::DYN_LORA_ALLOCATION_SCALE_DOWN_COOLDOWN_TICKS)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(defaults.scale_down_cooldown_ticks);

        let rate_window_multiplier = std::env::var(llm::DYN_LORA_ALLOCATION_RATE_WINDOW_MULTIPLIER)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.rate_window_multiplier);

        let buckets_per_second = std::env::var(llm::DYN_LORA_ALLOCATION_BUCKETS_PER_SECOND)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.buckets_per_second);

        let predictor_type = std::env::var(llm::DYN_LORA_ALLOCATION_PREDICTOR_TYPE)
            .ok()
            .and_then(|v| PredictorType::from_str(&v).ok())
            .unwrap_or(defaults.predictor_type);

        let ema_alpha = std::env::var(llm::DYN_LORA_ALLOCATION_EMA_ALPHA)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|a| a.clamp(0.0, 1.0))
            .unwrap_or(defaults.ema_alpha);

        let mcf = std::env::var(llm::DYN_LORA_MCF_CONFIG)
            .ok()
            .and_then(|v| {
                serde_json::from_str(&v)
                    .map_err(|e| {
                        tracing::warn!(
                            "Failed to parse DYN_LORA_MCF_CONFIG JSON, using defaults: {e}"
                        );
                        e
                    })
                    .ok()
            })
            .unwrap_or_default();

        Self {
            enabled,
            algorithm,
            timestep_secs,
            scale_down_cooldown_ticks,
            rate_window_multiplier,
            buckets_per_second,
            predictor_type,
            ema_alpha,
            mcf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LoraAllocationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.algorithm, AllocationAlgorithmType::Hrw);
        assert_eq!(config.timestep_secs, 3);
        assert_eq!(config.scale_down_cooldown_ticks, 3);
        assert_eq!(config.rate_window_multiplier, 30);
    }

    #[test]
    fn test_effective_rate_window_respects_minimum() {
        let config = LoraAllocationConfig::new(true, "hrw", 1, 2, 2).unwrap();
        assert_eq!(config.effective_rate_window_secs(), MIN_RATE_WINDOW_SECS);
    }

    #[test]
    fn test_effective_rate_window_uses_multiplier() {
        let config = LoraAllocationConfig::new(true, "hrw", 10, 2, 5).unwrap();
        assert_eq!(config.effective_rate_window_secs(), 50);
    }

    #[test]
    fn test_mcf_config_defaults() {
        let mcf = McfConfig::default();
        assert_eq!(mcf.candidate_m, 16);
        assert_eq!(mcf.alpha_pref, 1);
        assert_eq!(mcf.gamma_load, 1000);
        assert_eq!(mcf.beta_keep, 250);
        assert!(mcf.allow_overflow);
    }

    #[test]
    fn test_mcf_config_partial_json() {
        let json = r#"{"candidate_m":32,"gamma_load":2000}"#;
        let mcf: McfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(mcf.candidate_m, 32);
        assert_eq!(mcf.gamma_load, 2000);
        // Omitted fields use defaults
        assert_eq!(mcf.beta_keep, 250);
        assert_eq!(mcf.alpha_pref, 1);
    }

    #[test]
    fn test_mcf_algorithm_type_parsing() {
        assert_eq!(
            AllocationAlgorithmType::from_str("mcf").unwrap(),
            AllocationAlgorithmType::MinCostFlow
        );
        assert_eq!(
            AllocationAlgorithmType::from_str("min_cost_flow").unwrap(),
            AllocationAlgorithmType::MinCostFlow
        );
    }

    #[test]
    fn test_predictor_type_from_str() {
        assert_eq!(
            PredictorType::from_str("none").unwrap(),
            PredictorType::None
        );
        assert_eq!(PredictorType::from_str("ema").unwrap(), PredictorType::Ema);
        assert_eq!(PredictorType::from_str("EMA").unwrap(), PredictorType::Ema);
        assert!(PredictorType::from_str("invalid").is_err());
    }
}
