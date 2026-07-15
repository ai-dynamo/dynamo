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
    pub resume_timeout_seconds: f64,
    pub session_retention_seconds: f64,
    pub scheduler_interval_seconds: f64,
}

impl Default for SessionAwareAdmissionControlConfig {
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

impl SessionAwareAdmissionControlConfig {
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
        ] {
            assert!(config.validate().is_err());
        }
    }
}
