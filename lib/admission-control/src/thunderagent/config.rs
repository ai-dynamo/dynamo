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
    pub resume_timeout_seconds: f64,
    pub session_retention_seconds: f64,
    pub scheduler_interval_seconds: f64,
}

impl Default for ThunderAgentConfig {
    fn default() -> Self {
        Self {
            pause_threshold: 0.95,
            pause_target: 0.80,
            resume_timeout_seconds: 1_800.0,
            session_retention_seconds: 1_800.0,
            scheduler_interval_seconds: 5.0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_use_defaults_and_reject_unknown_fields() {
        let options =
            serde_yaml::from_str::<Mapping>("pause_threshold: 0.7\npause_target: 0.6").unwrap();
        let config = ThunderAgentConfig::from_options(&options).unwrap();
        assert_eq!(config.pause_threshold, 0.7);
        assert_eq!(config.pause_target, 0.6);
        assert_eq!(config.session_retention_seconds, 1_800.0);

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
                resume_timeout_seconds: 0.0,
                ..Default::default()
            },
            ThunderAgentConfig {
                session_retention_seconds: 0.0,
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
