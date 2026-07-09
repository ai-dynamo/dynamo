// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use serde::Deserialize;
use serde_yaml::{Mapping, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid ThunderAgent configuration: {0}")]
    Invalid(&'static str),
    #[error("failed to parse ThunderAgent configuration: {0}")]
    Parse(#[from] serde_yaml::Error),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThunderAgentConfig {
    pub pause_threshold: f64,
    pub pause_target: f64,
    pub resume_hysteresis: f64,
    pub resume_timeout_seconds: f64,
    pub scheduler_interval_seconds: f64,
    pub acting_token_weight: f64,
    pub acting_decay_tau_seconds: f64,
    pub buffer_per_program: usize,
}

impl Default for ThunderAgentConfig {
    fn default() -> Self {
        Self {
            pause_threshold: 0.95,
            pause_target: 0.80,
            resume_hysteresis: 0.10,
            resume_timeout_seconds: 1_800.0,
            scheduler_interval_seconds: 5.0,
            acting_token_weight: 1.0,
            acting_decay_tau_seconds: 1.0,
            buffer_per_program: 100,
        }
    }
}

impl ThunderAgentConfig {
    pub fn from_options(options: &Mapping) -> Result<Self, ConfigError> {
        let config: Self = serde_yaml::from_value(Value::Mapping(options.clone()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_fraction(self.pause_threshold, "pause_threshold must be in [0, 1]")?;
        validate_fraction(self.pause_target, "pause_target must be in [0, 1]")?;
        if self.pause_target > self.pause_threshold {
            return Err(ConfigError::Invalid(
                "pause_target must not exceed pause_threshold",
            ));
        }
        if !self.resume_hysteresis.is_finite()
            || self.resume_hysteresis < 0.0
            || self.resume_hysteresis > 1.0
        {
            return Err(ConfigError::Invalid("resume_hysteresis must be in [0, 1]"));
        }
        if self.resume_hysteresis > self.pause_threshold {
            return Err(ConfigError::Invalid(
                "resume_hysteresis must not exceed pause_threshold",
            ));
        }
        validate_positive_duration(
            self.resume_timeout_seconds,
            "resume_timeout_seconds must be a positive duration",
        )?;
        validate_positive_duration(
            self.scheduler_interval_seconds,
            "scheduler_interval_seconds must be a positive duration",
        )?;
        if !self.acting_token_weight.is_finite() || self.acting_token_weight <= 0.0 {
            return Err(ConfigError::Invalid(
                "acting_token_weight must be finite and positive",
            ));
        }
        if !self.acting_decay_tau_seconds.is_finite() || self.acting_decay_tau_seconds <= 0.0 {
            return Err(ConfigError::Invalid(
                "acting_decay_tau_seconds must be finite and positive",
            ));
        }
        Ok(())
    }
}

fn validate_fraction(value: f64, message: &'static str) -> Result<(), ConfigError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ConfigError::Invalid(message))
    }
}

fn validate_positive_duration(value: f64, message: &'static str) -> Result<(), ConfigError> {
    match Duration::try_from_secs_f64(value) {
        Ok(duration) if !duration.is_zero() => Ok(()),
        _ => Err(ConfigError::Invalid(message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_use_defaults_and_reject_unknown_fields() {
        let options = serde_yaml::from_str::<Mapping>(
            "pause_threshold: 0.7\npause_target: 0.6\nresume_hysteresis: 0.05",
        )
        .unwrap();
        let config = ThunderAgentConfig::from_options(&options).unwrap();
        assert_eq!(config.pause_threshold, 0.7);
        assert_eq!(config.pause_target, 0.6);

        let unknown = serde_yaml::from_str::<Mapping>("unknown: 1").unwrap();
        assert!(ThunderAgentConfig::from_options(&unknown).is_err());
    }

    #[test]
    fn rejects_inconsistent_or_zero_control_values() {
        for config in [
            ThunderAgentConfig {
                pause_threshold: 0.7,
                pause_target: 0.8,
                ..Default::default()
            },
            ThunderAgentConfig {
                resume_hysteresis: 0.96,
                ..Default::default()
            },
            ThunderAgentConfig {
                resume_timeout_seconds: 0.0,
                ..Default::default()
            },
            ThunderAgentConfig {
                acting_token_weight: 0.0,
                ..Default::default()
            },
            ThunderAgentConfig {
                scheduler_interval_seconds: 0.0,
                ..Default::default()
            },
            ThunderAgentConfig {
                scheduler_interval_seconds: 1e-20,
                ..Default::default()
            },
            ThunderAgentConfig {
                resume_timeout_seconds: 1e-20,
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err());
        }
    }
}
