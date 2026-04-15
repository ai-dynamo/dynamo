// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load prediction traits and implementations for smoothing per-LoRA load estimates.

use crate::lora::load_estimator::BucketedRateCounter;
use std::time::Instant;

/// Trait for load predictors that smooth or forecast per-LoRA load.
pub trait LoadPredictor: Send {
    fn update(&mut self, counter: &BucketedRateCounter, now: Instant);
    fn predict(&self) -> f64;
    fn reset(&mut self);
    fn name(&self) -> &'static str;
}

/// Exponential Moving Average predictor.
///
/// `estimate = alpha * measurement + (1 - alpha) * prev_estimate`
pub struct EmaPredictor {
    alpha: f64,
    estimate: Option<f64>,
}

impl EmaPredictor {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
            estimate: None,
        }
    }
}

impl LoadPredictor for EmaPredictor {
    fn update(&mut self, counter: &BucketedRateCounter, now: Instant) {
        let measurement = counter.count(now) as f64;
        self.estimate = Some(match self.estimate {
            Some(prev) => self.alpha * measurement + (1.0 - self.alpha) * prev,
            None => measurement,
        });
    }

    fn predict(&self) -> f64 {
        self.estimate.unwrap_or(0.0)
    }

    fn reset(&mut self) {
        self.estimate = None;
    }

    fn name(&self) -> &'static str {
        "ema"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_counter(count: u64) -> (BucketedRateCounter, Instant) {
        let now = Instant::now();
        let counter = BucketedRateCounter::new(30, Duration::from_secs(1), now);
        counter.record_count(count, now);
        (counter, now)
    }

    #[test]
    fn test_ema_first_observation() {
        let mut ema = EmaPredictor::new(0.3);
        let (counter, now) = make_counter(10);

        ema.update(&counter, now);
        assert!((ema.predict() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut ema = EmaPredictor::new(0.5);
        let now = Instant::now();
        let counter = BucketedRateCounter::new(30, Duration::from_secs(1), now);

        counter.record_count(10, now);
        ema.update(&counter, now);
        assert!((ema.predict() - 10.0).abs() < f64::EPSILON);

        // total = 20, EMA = 0.5 * 20 + 0.5 * 10 = 15
        counter.record_count(10, now);
        ema.update(&counter, now);
        assert!((ema.predict() - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ema_reset() {
        let mut ema = EmaPredictor::new(0.3);
        let (counter, now) = make_counter(10);

        ema.update(&counter, now);
        assert!(ema.predict() > 0.0);

        ema.reset();
        assert!((ema.predict()).abs() < f64::EPSILON);
    }
}
