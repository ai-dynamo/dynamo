// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionAwareConfig {
    pub pause_threshold: f64,
    pub pause_target: f64,
    pub resume_hysteresis: f64,
    pub resume_timeout_seconds: f64,
    pub resume_priority_boost: f64,
    pub scheduler_interval_seconds: f64,
    pub acting_token_weight: f64,
    pub acting_decay_tau_seconds: f64,
    pub buffer_per_program: usize,
}

impl Default for SessionAwareConfig {
    fn default() -> Self {
        Self {
            pause_threshold: 0.95,
            pause_target: 0.80,
            resume_hysteresis: 0.10,
            resume_timeout_seconds: 1800.0,
            resume_priority_boost: 1.0,
            scheduler_interval_seconds: 5.0,
            acting_token_weight: 1.0,
            acting_decay_tau_seconds: 1.0,
            buffer_per_program: 100,
        }
    }
}

impl SessionAwareConfig {
    pub(crate) fn validate(&self, location: &str) -> Result<(), String> {
        let fraction = |value: f64| value.is_finite() && (0.0..=1.0).contains(&value);
        if !fraction(self.pause_threshold) {
            return Err(invalid(
                location,
                "pause_threshold must be finite and in [0, 1]",
            ));
        }
        if !fraction(self.pause_target) || self.pause_target > self.pause_threshold {
            return Err(invalid(
                location,
                "pause_target must be finite and in [0, pause_threshold]",
            ));
        }
        if !fraction(self.resume_hysteresis) || self.resume_hysteresis > self.pause_threshold {
            return Err(invalid(
                location,
                "resume_hysteresis must be finite and in [0, pause_threshold]",
            ));
        }
        for (name, value) in [
            ("resume_timeout_seconds", self.resume_timeout_seconds),
            (
                "scheduler_interval_seconds",
                self.scheduler_interval_seconds,
            ),
            ("acting_token_weight", self.acting_token_weight),
            ("acting_decay_tau_seconds", self.acting_decay_tau_seconds),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(location, &format!("{name} must be finite and > 0")));
            }
        }
        if !self.resume_priority_boost.is_finite() || self.resume_priority_boost < 0.0 {
            return Err(invalid(
                location,
                "resume_priority_boost must be finite and >= 0",
            ));
        }
        Ok(())
    }

    pub fn scheduler_interval(&self) -> Duration {
        Duration::from_secs_f64(self.scheduler_interval_seconds)
    }
}

fn invalid(location: &str, message: &str) -> String {
    format!("{location} queue_admission {message}")
}
