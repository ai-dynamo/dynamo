// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompassConfig {
    pub weights: WeightProfile,
    pub sampling_rate: f64,
    pub confidence_thresholds: ConfidenceThresholds,
    pub floor_params: FloorParams,
    pub calibration_threshold_pct: f64,
}

impl Default for CompassConfig {
    fn default() -> Self {
        Self {
            weights: WeightProfile::default(),
            sampling_rate: 0.01,
            confidence_thresholds: ConfidenceThresholds::default(),
            floor_params: FloorParams::default(),
            calibration_threshold_pct: 15.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightProfile {
    pub name: String,
    pub critical_path_weight: f64,
    pub saturation_weight: f64,
    pub floor_ratio_weight: f64,
}

impl Default for WeightProfile {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            critical_path_weight: 0.6,
            saturation_weight: 0.3,
            floor_ratio_weight: 0.1,
        }
    }
}

impl WeightProfile {
    pub fn conservative() -> Self {
        Self {
            name: "conservative".to_string(),
            critical_path_weight: 0.8,
            saturation_weight: 0.15,
            floor_ratio_weight: 0.05,
        }
    }

    pub fn aggressive() -> Self {
        Self {
            name: "aggressive".to_string(),
            critical_path_weight: 0.4,
            saturation_weight: 0.4,
            floor_ratio_weight: 0.2,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "default" => Some(Self::default()),
            "conservative" => Some(Self::conservative()),
            "aggressive" => Some(Self::aggressive()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceThresholds {
    pub high_gap: f64,
    pub medium_gap: f64,
}

impl Default for ConfidenceThresholds {
    fn default() -> Self {
        Self {
            high_gap: 0.20,
            medium_gap: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorParams {
    pub pointer_chase_ns: f64,
    pub optimization_candidate_threshold: f64,
}

impl Default for FloorParams {
    fn default() -> Self {
        Self {
            pointer_chase_ns: 80.0,
            optimization_candidate_threshold: 2.0,
        }
    }
}

pub const COMPASS_VERSION: &str = "1.0.0";

pub fn probes_enabled() -> bool {
    std::env::var("DYN_COMPASS_PROBES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_weights_sum_to_one() {
        let w = WeightProfile::default();
        let sum = w.critical_path_weight + w.saturation_weight + w.floor_ratio_weight;
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weight_profiles() {
        assert!(WeightProfile::from_name("default").is_some());
        assert!(WeightProfile::from_name("conservative").is_some());
        assert!(WeightProfile::from_name("aggressive").is_some());
        assert!(WeightProfile::from_name("unknown").is_none());
    }

    #[test]
    fn test_config_serialization() {
        let config = CompassConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CompassConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.weights.name, "default");
    }
}
