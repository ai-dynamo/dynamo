// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicU64, Ordering};

pub struct SamplingController {
    counter: AtomicU64,
    rate_inverse: u64,
}

impl SamplingController {
    pub fn new(rate: f64) -> Self {
        let rate_inverse = if rate > 0.0 && rate <= 1.0 {
            (1.0 / rate) as u64
        } else {
            1
        };
        Self {
            counter: AtomicU64::new(0),
            rate_inverse,
        }
    }

    pub fn should_sample(&self) -> bool {
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        count % self.rate_inverse == 0
    }
}

impl Default for SamplingController {
    fn default() -> Self {
        let rate: f64 = std::env::var("DYN_COMPASS_SAMPLE_RATE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.01);
        Self::new(rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_100_percent_sampling() {
        let sc = SamplingController::new(1.0);
        for _ in 0..100 {
            assert!(sc.should_sample());
        }
    }

    #[test]
    fn test_10_percent_sampling() {
        let sc = SamplingController::new(0.1);
        let sampled: usize = (0..1000).filter(|_| sc.should_sample()).count();
        assert!(sampled >= 90 && sampled <= 110);
    }

    #[test]
    fn test_1_percent_sampling() {
        let sc = SamplingController::new(0.01);
        let sampled: usize = (0..10000).filter(|_| sc.should_sample()).count();
        assert!(sampled >= 90 && sampled <= 110);
    }
}
