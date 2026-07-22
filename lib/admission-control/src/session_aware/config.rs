// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid session-aware admission-control configuration: {0}")]
    Invalid(&'static str),
    #[error("failed to parse session-aware admission-control configuration: {0}")]
    Parse(#[from] serde_yaml::Error),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionAwareAdmissionControlConfig {
    pub pause_threshold: f64,
    pub pause_target: f64,
    pub resume_hysteresis: f64,
    pub resume_timeout_seconds: f64,
    pub session_retention_seconds: f64,
    pub scheduler_interval_seconds: f64,
    pub acting_token_weight: f64,
    pub acting_decay_tau_seconds: f64,
    pub buffer_per_program: usize,
}

impl Default for SessionAwareAdmissionControlConfig {
    fn default() -> Self {
        Self {
            pause_threshold: 0.95,
            pause_target: 0.80,
            resume_hysteresis: 0.10,
            resume_timeout_seconds: 1_800.0,
            session_retention_seconds: 1_800.0,
            scheduler_interval_seconds: 5.0,
            acting_token_weight: 1.0,
            acting_decay_tau_seconds: 1.0,
            buffer_per_program: 100,
        }
    }
}

impl SessionAwareAdmissionControlConfig {
    pub fn from_options(options: &serde_yaml::Mapping) -> Result<Self, ConfigError> {
        let config: Self = serde_yaml::from_value(serde_yaml::Value::Mapping(options.clone()))?;
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
        validate_fraction(
            self.resume_hysteresis,
            "resume_hysteresis must be in [0, 1]",
        )?;
        if self.resume_hysteresis > self.pause_threshold {
            return Err(ConfigError::Invalid(
                "resume_hysteresis must not exceed pause_threshold",
            ));
        }
        validate_positive_number(
            self.acting_token_weight,
            "acting_token_weight must be positive",
        )?;
        validate_positive_number(
            self.acting_decay_tau_seconds,
            "acting_decay_tau_seconds must be positive",
        )?;
        validate_positive_duration(
            self.resume_timeout_seconds,
            "resume_timeout_seconds must be a positive duration",
        )?;
        validate_positive_duration(
            self.session_retention_seconds,
            "session_retention_seconds must be a positive duration",
        )?;
        validate_positive_duration(
            self.scheduler_interval_seconds,
            "scheduler_interval_seconds must be a positive duration",
        )?;
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

fn validate_positive_number(value: f64, message: &'static str) -> Result<(), ConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ConfigError::Invalid(message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_python_scheduler_knobs_with_other_defaults() {
        let options = serde_yaml::from_str(
            "scheduler_interval_seconds: 3.0\nresume_hysteresis: 0.05\nacting_token_weight: 0.75\nacting_decay_tau_seconds: 2.0\nbuffer_per_program: 64\n",
        )
        .unwrap();
        let config = SessionAwareAdmissionControlConfig::from_options(&options).unwrap();

        assert_eq!(config.scheduler_interval_seconds, 3.0);
        assert_eq!(config.pause_threshold, 0.95);
        assert_eq!(config.resume_hysteresis, 0.05);
        assert_eq!(config.acting_token_weight, 0.75);
        assert_eq!(config.acting_decay_tau_seconds, 2.0);
        assert_eq!(config.buffer_per_program, 64);
    }

    #[test]
    fn rejects_unknown_policy_options() {
        let options = serde_yaml::from_str("unknown: true\n").unwrap();

        assert!(SessionAwareAdmissionControlConfig::from_options(&options).is_err());
    }

    #[test]
    fn rejects_inconsistent_or_zero_control_values() {
        for config in [
            SessionAwareAdmissionControlConfig {
                pause_threshold: 0.7,
                pause_target: 0.8,
                ..Default::default()
            },
            SessionAwareAdmissionControlConfig {
                resume_timeout_seconds: 0.0,
                ..Default::default()
            },
            SessionAwareAdmissionControlConfig {
                session_retention_seconds: 0.0,
                ..Default::default()
            },
            SessionAwareAdmissionControlConfig {
                scheduler_interval_seconds: 0.0,
                ..Default::default()
            },
            SessionAwareAdmissionControlConfig {
                scheduler_interval_seconds: 1e-20,
                ..Default::default()
            },
            SessionAwareAdmissionControlConfig {
                resume_timeout_seconds: 1e-20,
                ..Default::default()
            },
            SessionAwareAdmissionControlConfig {
                pause_threshold: 0.7,
                resume_hysteresis: 0.8,
                ..Default::default()
            },
            SessionAwareAdmissionControlConfig {
                acting_token_weight: 0.0,
                ..Default::default()
            },
            SessionAwareAdmissionControlConfig {
                acting_decay_tau_seconds: 0.0,
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err());
        }
    }
}
