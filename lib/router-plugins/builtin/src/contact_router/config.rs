// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;

use crate::contact_router::group_priority::PriorityConfig;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("invalid ContactRouter configuration: {0}")]
    Invalid(&'static str),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContactRouterConfig {
    pub pause_threshold: f64,
    pub pause_target: f64,
    pub resume_hysteresis: f64,
    pub resume_timeout_seconds: f64,
    pub session_retention_seconds: f64,
    pub scheduler_interval_seconds: f64,
    pub acting_token_weight: f64,
    pub acting_decay_tau_seconds: f64,
    pub buffer_per_program: usize,
    pub max_tracked_requests: usize,
    /// Pause the lowest group score first and resume the highest first.
    pub priority: Option<PriorityConfig>,
    /// Adapt how many programs may run so groups finish this many versions
    /// old; programs of groups not yet started wait without a version.
    pub target_age: Option<f64>,
    /// Programs the adaptive limit starts at.
    pub initial_programs: usize,
    /// Programs added to the limit per group program, per version a group
    /// finishes younger than `target_age` (removed when older).
    pub age_gain: f64,
    /// Do not move the limit until the current weight version passes this.
    pub age_delay_versions: u64,
    /// Start a new group only if every started program, at the average final
    /// context of finished programs, fits under `pause_threshold` of KV.
    pub kv_admission: bool,
}

impl Default for ContactRouterConfig {
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
            max_tracked_requests: 10_000,
            priority: None,
            target_age: None,
            initial_programs: 256,
            age_gain: 0.1,
            age_delay_versions: 0,
            kv_admission: false,
        }
    }
}

impl ContactRouterConfig {
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
        validate_duration(
            self.resume_timeout_seconds,
            "resume_timeout_seconds must be positive",
        )?;
        validate_duration(
            self.session_retention_seconds,
            "session_retention_seconds must be positive",
        )?;
        validate_duration(
            self.scheduler_interval_seconds,
            "scheduler_interval_seconds must be positive",
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
        if self.max_tracked_requests == 0 {
            return Err(ConfigError::Invalid(
                "max_tracked_requests must be positive",
            ));
        }
        Ok(())
    }

    pub(crate) fn resume_timeout(&self) -> Duration {
        Duration::from_secs_f64(self.resume_timeout_seconds)
    }

    pub(crate) fn session_retention(&self) -> Duration {
        Duration::from_secs_f64(self.session_retention_seconds)
    }

    pub(crate) fn scheduler_interval(&self) -> Duration {
        Duration::from_secs_f64(self.scheduler_interval_seconds)
    }
}

fn validate_fraction(value: f64, message: &'static str) -> Result<(), ConfigError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ConfigError::Invalid(message))
    }
}

fn validate_duration(value: f64, message: &'static str) -> Result<(), ConfigError> {
    match Duration::try_from_secs_f64(value) {
        Ok(duration) if !duration.is_zero() => Ok(()),
        _ => Err(ConfigError::Invalid(message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_control_values() {
        assert!(ContactRouterConfig::default().validate().is_ok());
        assert!(
            ContactRouterConfig {
                pause_target: 0.96,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ContactRouterConfig {
                scheduler_interval_seconds: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
