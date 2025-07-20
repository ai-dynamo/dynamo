// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rate limiter implementation for the OpenAI API compatible HTTP service.
//!
//! The rate limiter is used to limit the rate of requests to the HTTP service.
//! It is used to prevent abuse of the service, under heavy load,
//! and to ensure that the service is available to all users. The system values
//! 'good-put' (that is, the throughput of the system under good performance metrics, in the form of
//! time to first token and inter-token latency) over 'throughput' (that is, the total amount of
//! tokens processed), and the rate limiter is used to ensure that the service is available to all
//! users, even under heavy load.
//!
//! The rate limiter is implemented using a time-weighted exponential moving average (EMA).
//! The time-weighted average is computed using the following formula:
//!
//! ```text
//! average = sum(value * weight) / sum(weight)
//! ```
//!
//! Where `weight` is the weight of the sample based on the age of the sample and the time constant:
//!
//! ```text
//! age = now - record_time
//! weight = exp(-age / time_constant_secs)
//! ```
//!
//! Where `now` is the current time, `record_time` is the time the sample was recorded,
//! and `time_constant_secs` is the time constant for the time-weighted average.
//!
//! Moreover, we decay the average to account for the time elapsed since the last update.
//! This models "system recovery" during idle time. This is done by multiplying the average by the
//! decay factor:
//!
//! ```text
//! decayed_average = average * exp(-time_elapsed / time_constant_secs)
//! ```

use std::time::Instant;

use anyhow::Result;
use dashmap::DashMap;
use derive_builder::Builder;
use validator::Validate;

/// Configuration for the rate limiter
#[derive(Debug, Clone, Builder, Validate)]
#[builder(pattern = "owned")]
pub struct RateLimiterConfig {
    /// Threshold for the time to first token metric,
    /// which defines the maximum allowed time to first token
    /// in seconds. Any recorded time to first token above this threshold
    /// will likely trigger a rate limit rejection for the next incoming request.
    #[builder(default = "1.0")]
    #[validate(range(min = 1e-2))]
    ttft_threshold_secs: f64,

    /// Threshold for the inter-token latency metric,
    /// which defines the maximum allowed inter-token latency
    /// in seconds. Any recorded inter-token latency above this threshold
    /// will likely trigger a rate limit rejection for the next incoming request.
    #[builder(default = "0.1")]
    #[validate(range(min = 1e-4))]
    itl_threshold_secs: f64,

    /// Time constant for the time-weighted EMA,
    /// that is, the time constant for the exponential moving average
    /// of the time-weighted average.
    #[builder(default = "15.0")]
    #[validate(range(min = 1e-2))]
    time_constant_secs: f64,

    /// Whether to use per-model limits, that is,
    /// to track rate limit metrics for each model separately
    #[builder(default = "true")]
    per_model_limits: bool,

    /// Whether the rate limiter is enabled
    #[builder(default = "false")]
    is_enabled: bool,
}

impl RateLimiterConfig {
    pub fn new(
        ttft_threshold_secs: f64,
        itl_threshold_secs: f64,
        time_constant_secs: f64,
        per_model_limits: bool,
    ) -> Result<Self> {
        let config: RateLimiterConfig = Self {
            ttft_threshold_secs,
            itl_threshold_secs,
            time_constant_secs,
            per_model_limits,
            is_enabled: true,
        };

        config
            .validate()
            .map_err(|e| anyhow::anyhow!("Invalid rate limiter config: {}", e))?;

        Ok(config)
    }

    pub fn empty() -> Self {
        Self {
            ttft_threshold_secs: 0.0,
            itl_threshold_secs: 0.0,
            time_constant_secs: 0.001,
            per_model_limits: false,
            is_enabled: false,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.is_enabled
    }

    pub fn builder() -> RateLimiterConfigBuilder {
        RateLimiterConfigBuilder::default()
    }
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            ttft_threshold_secs: 1.0, // 1s
            itl_threshold_secs: 0.1,  // 100ms
            time_constant_secs: 30.0, // 30s
            per_model_limits: false,
            is_enabled: true,
        }
    }
}

/// Tracks recent samples to compute time-weighted averages of a metric. Formally,
/// the time-weighted average is defined as:
///
/// ```text
/// average = sum(value * weight) / sum(weight)
/// ```
///
/// Where `weight` is the weight of the sample based on the age of the sample and the time constant:
///
/// ```text
/// age = now - record_time
/// weight = exp(-age / time_constant_secs)
/// ```
///
/// Where `now` is the current time, `record_time` is the time the sample was recorded,
/// and `time_constant_secs` is the time constant for the time-weighted average.
/// In this way, more recent samples have a higher weight than older samples, the latter
/// decaying exponentially towards zero (making it less impactful for the current average calculation).
///
/// In order to compute the time-weighted average more efficiently, we leverage the well
/// known property of the exponential function:
///
/// ```text
/// exp(x) = exp(y) * exp(x - y)
/// ```
///
/// This allows us to compute the time-weighted average, recursively, in a single pass,
/// (see Markov's property) as follows:
///
/// ```text
/// previous_weight_total = sum(weight)
/// updated_factor = 1 / (1 + previous_weight_total * exp(-age / time_constant_secs))
/// average(now) = average(last_time) * updated_factor + value * (1 - updated_factor)
/// ```
#[derive(Debug)]
pub struct TimeWeightedAverageTracker {
    previous_weighted_average: f64,
    previous_total_weight: f64,
    previous_observed_time: Instant,
    time_constant_secs: f64,
}

impl TimeWeightedAverageTracker {
    pub fn new(time_constant_secs: f64) -> Self {
        let now = Instant::now();
        Self {
            previous_weighted_average: 0.,
            previous_total_weight: 0.,
            previous_observed_time: now,
            time_constant_secs,
        }
    }

    /// Record a new value to the tracker.
    pub fn record_value(&mut self, value: f64) {
        let now = Instant::now();
        if self.previous_weighted_average == 0. && self.previous_total_weight == 0. {
            // First sample
            self.previous_weighted_average = value;
            self.previous_total_weight = 1.;
        } else {
            let time_elapsed = now
                .duration_since(self.previous_observed_time)
                .as_secs_f64();
            let decay_factor = (-time_elapsed / self.time_constant_secs).exp();

            // Update the weighted average, using recursive EMA formula
            self.previous_total_weight = 1. + self.previous_total_weight * decay_factor;
            let alpha = 1. / self.previous_total_weight;
            self.previous_weighted_average =
                alpha * value + (1. - alpha) * self.previous_weighted_average;
        }

        self.previous_observed_time = now;
    }

    /// Get the current time-weighted average, decayed to account for the time elapsed since the last update.
    pub fn get_decayed_time_weighted_average(&self) -> f64 {
        let now = Instant::now();
        let time_elapsed = now
            .duration_since(self.previous_observed_time)
            .as_secs_f64();
        let decay_factor = (-time_elapsed / self.time_constant_secs).exp();
        self.previous_weighted_average * decay_factor
    }
}

#[derive(Debug)]
struct ModelMetrics {
    ttft_tracker: TimeWeightedAverageTracker,
    itl_tracker: TimeWeightedAverageTracker,
}

impl ModelMetrics {
    fn new(config: &RateLimiterConfig) -> Self {
        let ttft_tracker = TimeWeightedAverageTracker::new(config.time_constant_secs);
        let itl_tracker = TimeWeightedAverageTracker::new(config.time_constant_secs);

        Self {
            ttft_tracker,
            itl_tracker,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RateLimiterMetrics {
    pub ttft_diagnostics: TimeWeightedDiagnostics,
    pub itl_diagnostics: TimeWeightedDiagnostics,
}

impl std::fmt::Display for RateLimiterMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RateLimiterMetrics {{\n  TTFT: {},\n  ITL:  {}\n}}",
            self.ttft_diagnostics, self.itl_diagnostics
        )
    }
}

#[derive(Debug, Clone)]
pub struct TimeWeightedDiagnostics {
    pub decayed_time_weighted_average: f64,
    pub time_constant_secs: f64,
    pub previous_weighted_sum: f64,
    pub previous_observed_time: Instant,
}

impl std::fmt::Display for TimeWeightedDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TimeWeightedDiagnostics {{ \
                decayed_time_weighted_average: {:.3}, \
                time_constant_secs: {:.1}, \
                previous_weighted_sum: {:.3}, \
                duration_since_last_update: {:?} \
            }}",
            self.decayed_time_weighted_average,
            self.time_constant_secs,
            self.previous_weighted_sum,
            self.previous_observed_time.elapsed().as_secs_f64()
        )
    }
}

pub struct RateLimiter {
    config: RateLimiterConfig,
    model_metrics: DashMap<String, ModelMetrics>,
}

impl RateLimiter {
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            config,
            model_metrics: DashMap::new(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.is_enabled
    }

    #[inline]
    fn get_model_key(&self, model: &str) -> String {
        if self.config.per_model_limits {
            model.to_string()
        } else {
            "global".to_string()
        }
    }

    /// Record the time to first token metric for a given model
    pub fn record_ttft(&self, model: &str, ttft_ms: f64) {
        let model_key = self.get_model_key(model);
        let mut model_metrics = self
            .model_metrics
            .entry(model_key)
            .or_insert_with(|| ModelMetrics::new(&self.config));

        model_metrics.ttft_tracker.record_value(ttft_ms);
    }

    /// Record the inter-token latency metric for a given model
    pub fn record_itl(&self, model: &str, itl_ms: f64) {
        let model_key = self.get_model_key(model);
        let mut model_metrics = self
            .model_metrics
            .entry(model_key)
            .or_insert_with(|| ModelMetrics::new(&self.config));

        model_metrics.itl_tracker.record_value(itl_ms);
    }

    /// Check if the request should be rejected based on the cached metrics
    ///
    /// Returns true if the request should be rejected, false otherwise
    pub fn should_reject(&self, model: &str) -> ShouldRejectResult {
        let model_key = self.get_model_key(model);
        let model_metrics = self.model_metrics.get(&model_key);

        let Some(model_metrics) = model_metrics else {
            return ShouldRejectResult {
                should_reject: false,
                decayed_ttft_ema: 0.0,
                decayed_itl_ema: 0.0,
            };
        };

        // Get decayed time-weighted EMA values
        let decayed_ttft_ema = model_metrics
            .ttft_tracker
            .get_decayed_time_weighted_average();
        let decayed_itl_ema = model_metrics
            .itl_tracker
            .get_decayed_time_weighted_average();

        drop(model_metrics);

        let ttft_exceeded = self.config.ttft_threshold_secs < decayed_ttft_ema;
        let itl_exceeded = self.config.itl_threshold_secs < decayed_itl_ema;

        if ttft_exceeded || itl_exceeded {
            let rate_limiter_metrics = self.get_metrics(&model_key);
            self.log_metrics(model, rate_limiter_metrics, true);
            return ShouldRejectResult {
                should_reject: true,
                decayed_ttft_ema,
                decayed_itl_ema,
            };
        }

        if decayed_ttft_ema > self.config.ttft_threshold_secs * 0.9
            || decayed_itl_ema > self.config.itl_threshold_secs * 0.9
        {
            let rate_limiter_metrics = self.get_metrics(&model_key);
            self.log_metrics(model, rate_limiter_metrics, false);
        }

        ShouldRejectResult {
            should_reject: false,
            decayed_ttft_ema,
            decayed_itl_ema,
        }
    }

    /// Get current metrics and diagnostics for current model
    #[inline]
    fn get_metrics(&self, model_key: &str) -> RateLimiterMetrics {
        let model_metrics = self.model_metrics.get(model_key).unwrap();
        let decayed_ttft_ema = model_metrics
            .ttft_tracker
            .get_decayed_time_weighted_average();
        let decayed_itl_ema = model_metrics
            .itl_tracker
            .get_decayed_time_weighted_average();
        let ttft_previous_weighted_sum = model_metrics.ttft_tracker.previous_total_weight;
        let itl_previous_weighted_sum = model_metrics.itl_tracker.previous_total_weight;
        let ttft_previous_observed_time = model_metrics.ttft_tracker.previous_observed_time;
        let itl_previous_observed_time = model_metrics.itl_tracker.previous_observed_time;

        RateLimiterMetrics {
            ttft_diagnostics: TimeWeightedDiagnostics {
                decayed_time_weighted_average: decayed_ttft_ema,
                time_constant_secs: self.config.time_constant_secs,
                previous_weighted_sum: ttft_previous_weighted_sum,
                previous_observed_time: ttft_previous_observed_time,
            },
            itl_diagnostics: TimeWeightedDiagnostics {
                decayed_time_weighted_average: decayed_itl_ema,
                time_constant_secs: self.config.time_constant_secs,
                previous_weighted_sum: itl_previous_weighted_sum,
                previous_observed_time: itl_previous_observed_time,
            },
        }
    }

    fn log_metrics(&self, model: &str, metrics: RateLimiterMetrics, has_exceeded: bool) {
        if has_exceeded {
            tracing::warn!(
                model = model,
                ttft_threshold_secs = self.config.ttft_threshold_secs,
                itl_threshold_secs = self.config.itl_threshold_secs,
                "Rate limit exceeded for model {model}: {metrics}",
                metrics = metrics,
            );
        } else {
            tracing::info!(
                model = model,
                ttft_threshold_secs = self.config.ttft_threshold_secs,
                itl_threshold_secs = self.config.itl_threshold_secs,
                "Approaching rate limit thresholds. Current rate limit metrics for model {model}: {metrics}",
                metrics = metrics,
            );
        }
    }
}

pub struct ShouldRejectResult {
    pub should_reject: bool,
    pub decayed_ttft_ema: f64,
    pub decayed_itl_ema: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::{atomic::AtomicUsize, Arc};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_simple_time_weighted_average_tracker() {
        const TIME_CONSTANT_SECS: f64 = 1.0; // Short time constant

        const SLEEP_DURATION_MS: u64 = 100;

        let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

        // Add samples with increasing delays
        tracker.record_value(100.0);
        thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));
        tracker.record_value(200.0);
        thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));
        tracker.record_value(300.0);
        thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));
        tracker.record_value(400.0);
        thread::sleep(Duration::from_millis(20 * SLEEP_DURATION_MS)); // Long gap
        tracker.record_value(500.0);

        let avg = tracker.get_decayed_time_weighted_average();
        assert!(avg > 0.0, "Average should be positive");
    }

    #[test]
    fn test_edge_case_all_samples_below_threshold() {
        const TIME_CONSTANT_SECS: f64 = 0.1;
        const SLEEP_DURATION_MS: u64 = 2_000;
        const EPSILON: f64 = 1e-5;

        let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

        tracker.record_value(100.0);
        thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));

        let avg = tracker.get_decayed_time_weighted_average();

        // Should return a close to 0.0 when time constant is small and time passed is large
        assert!(avg < EPSILON, "Average should be 0.0: {}", avg);
    }

    #[test]
    fn test_edge_case_single_sample() {
        const TIME_CONSTANT_SECS: f64 = 10.;
        const EPSILON: f64 = 0.5; // exp(-0.01) ~= 0.99

        let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);
        tracker.record_value(42.);
        thread::sleep(Duration::from_millis(100));

        let avg = tracker.get_decayed_time_weighted_average();

        assert!(
            (avg - 42.).abs() < EPSILON,
            "Average should be close to 42.0: {}",
            avg
        );
    }

    #[test]
    fn test_time_weighted_average_tracker_correctness() {
        const TIME_CONSTANT_SECS: f64 = 10.0;
        const NUM_SAMPLES: usize = 100;
        const SLEEP_DURATION_MS: u64 = 1;

        let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

        // Record old sample
        tracker.record_value(1000.0);
        thread::sleep(Duration::from_millis(1_000 * SLEEP_DURATION_MS));

        // Add more recent samples with lower values
        for _ in 0..NUM_SAMPLES {
            tracker.record_value(100.0);
            thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));
        }

        let avg = tracker.get_decayed_time_weighted_average();
        assert!(
            avg < 500.0,
            "Average should be dominated by recent samples: {}",
            avg
        );
        assert!(
            avg > 100.0,
            "Average should still be influenced by old sample: {}",
            avg
        );
    }

    #[test]
    fn test_time_weighted_average_quantitative_analysis() {
        const TIME_CONSTANT_SECS: f64 = 2.0; // 2 second time constant
        const EPSILON: f64 = 0.05; // 5% tolerance for timing precision
        const SLEEP_DURATION_MS: u64 = 100; // 100ms

        let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

        // Record samples with known values and controlled timing
        let sample_values = [100.0, 200.0, 300.0, 400.0];
        let sample_delays_ms = [0, 500, 1000, 1500]; // Delays in milliseconds

        let start_time = Instant::now();

        // Record first sample immediately
        tracker.record_value(sample_values[0]);

        // Record subsequent samples with known delays
        for i in 1..sample_values.len() {
            thread::sleep(Duration::from_millis(
                sample_delays_ms[i] - sample_delays_ms[i - 1],
            ));
            tracker.record_value(sample_values[i]);
        }

        // Wait a bit more, then calculate
        thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));
        let calculation_time = Instant::now();

        // Calculate expected weighted average manually
        let total_elapsed = calculation_time.duration_since(start_time);
        let mut expected_weighted_sum = 0.0;
        let mut expected_total_weight = 0.0;

        for i in 0..sample_values.len() {
            // Age of this sample = total_elapsed - delay_when_recorded
            let sample_age_secs =
                total_elapsed.as_secs_f64() - (sample_delays_ms[i] as f64 / 1000.0);
            let weight = f64::exp(-sample_age_secs / TIME_CONSTANT_SECS);

            expected_weighted_sum += sample_values[i] * weight;
            expected_total_weight += weight;

            println!(
                "Sample {}: value={}, age={:.3}s, weight={:.6}",
                i, sample_values[i], sample_age_secs, weight
            );
        }

        let expected_average =
            (expected_weighted_sum / expected_total_weight) * f64::exp(-0.1 / TIME_CONSTANT_SECS); // 0.1s is the time elapsed since the last sample
        let actual_average = tracker.get_decayed_time_weighted_average();

        println!("Expected average: {:.6}", expected_average);
        println!("Actual average: {:.6}", actual_average);
        println!(
            "Difference: {:.6}",
            (actual_average - expected_average).abs()
        );
        println!(
            "Relative error: {:.4}%",
            100.0 * (actual_average - expected_average).abs() / expected_average
        );

        // Verify the calculation is mathematically correct within tolerance
        let relative_error = (actual_average - expected_average).abs() / expected_average;
        assert!(
            relative_error < EPSILON,
            "Time-weighted average calculation error too large: expected {:.6}, got {:.6}, relative error {:.4}%",
            expected_average, actual_average, relative_error * 100.0
        );

        // Additional verification: more recent samples should have higher influence
        // Sample 3 (400.0) is most recent, so if we compare with a simple average:
        let simple_average = sample_values.iter().sum::<f64>() / sample_values.len() as f64;
        println!("Simple average: {:.6}", simple_average);

        // The time-weighted average should be closer to the most recent value (400.0)
        // than the simple average, since recent samples have higher weights
        let distance_to_recent = (actual_average - 400.0).abs();
        let distance_simple_to_recent = (simple_average - 400.0).abs();

        assert!(
            distance_to_recent < distance_simple_to_recent,
            "Time-weighted average should be closer to recent values than simple average: \
             weighted_avg={:.2}, simple_avg={:.2}, recent_value=400.0",
            actual_average,
            simple_average
        );
    }

    #[test]
    fn test_exponential_decay_verification() {
        const TIME_CONSTANT_SECS: f64 = 1.0; // 1 second time constant
        const EPSILON: f64 = 0.02; // 2% tolerance

        let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

        // Record a high value, then wait exactly one time constant
        tracker.record_value(1000.0);
        thread::sleep(Duration::from_millis(1_000)); // Wait 1 second = 1 time constant

        // Record a low value
        tracker.record_value(100.0);

        let actual_average = tracker.get_decayed_time_weighted_average();

        // After 1 time constant, the old sample should have weight = e^(-1) ≈ 0.368
        // New sample has weight ≈ 1.0
        let old_weight = f64::exp(-1.0); // ≈ 0.368
        let new_weight = 1.0;

        let expected_average =
            (1000.0 * old_weight + 100.0 * new_weight) / (old_weight + new_weight);

        println!("Old weight (e^-1): {:.6}", old_weight);
        println!("New weight: {:.6}", new_weight);
        println!("Expected average: {:.6}", expected_average);
        println!("Actual average: {:.6}", actual_average);

        let relative_error = (actual_average - expected_average).abs() / expected_average;
        assert!(
            relative_error < EPSILON,
            "Exponential decay verification failed: expected {:.6}, got {:.6}, error {:.4}%",
            expected_average,
            actual_average,
            relative_error * 100.0
        );

        // Verify the theoretical calculation: should be around 463.4
        assert!(
            (expected_average - 342.04727).abs() < 1e-5,
            "Theoretical calculation seems wrong: {:.1}",
            expected_average
        );
    }

    #[test]
    fn test_mathematical_properties() {
        const TIME_CONSTANT_SECS: f64 = 2.0;

        let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

        // Property 1: Single sample should return its own value
        tracker.record_value(42.0);
        let single_avg = tracker.get_decayed_time_weighted_average();
        assert!(
            (single_avg - 42.0).abs() < 1e-6,
            "Single sample average should equal sample value: {}",
            single_avg
        );

        // Property 2: Adding identical samples should not change average
        tracker.record_value(42.0);
        tracker.record_value(42.0);
        let identical_avg = tracker.get_decayed_time_weighted_average();
        assert!(
            (identical_avg - 42.0).abs() < 1e-5,
            "Identical samples should maintain average: {}",
            identical_avg
        );

        // Property 3: Average should be bounded by min and max values
        let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);
        let values = vec![10.0, 50.0, 30.0, 70.0, 20.0];
        let min_val = 10.0;
        let max_val = 70.0;

        for &val in &values {
            tracker.record_value(val);
            thread::sleep(Duration::from_millis(10));
        }

        let bounded_avg = tracker.get_decayed_time_weighted_average();
        assert!(
            bounded_avg >= min_val && bounded_avg <= max_val,
            "Average should be bounded: {:.2} not in [{:.2}, {:.2}]",
            bounded_avg,
            min_val,
            max_val
        );

        println!("Values: {:?}", values);
        println!(
            "Average: {:.2} ∈ [{:.2}, {:.2}] ✓",
            bounded_avg, min_val, max_val
        );
    }

    #[test]
    fn test_concurrent_access_simulation() {
        const NUM_THREADS: usize = 10;
        const NUM_RECORDS: usize = 100;

        const SLEEP_INTERVAL: usize = 10;
        const SLEEP_DURATION_MS: u64 = 1;

        let config = RateLimiterConfig {
            ttft_threshold_secs: 1.0,
            itl_threshold_secs: 0.1,
            time_constant_secs: 30.0,
            per_model_limits: false,
            is_enabled: true,
        };
        let limiter = Arc::new(RateLimiter::new(config));
        let error_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();

        for i in 0..NUM_THREADS {
            let limiter_clone = limiter.clone();
            let error_count_clone = error_count.clone();

            handles.push(thread::spawn(move || {
                for j in 0..NUM_RECORDS {
                    limiter_clone.record_ttft("model", (i * NUM_RECORDS + j) as f64 / 10_000.0);
                    limiter_clone.record_itl("model", (i + j) as f64 / 1_000.0);

                    if limiter_clone.should_reject("model").should_reject {
                        error_count_clone.fetch_add(1, Ordering::Relaxed);
                    }

                    if j % SLEEP_INTERVAL == 0 {
                        thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have no errors
        let error_count = error_count.load(Ordering::Relaxed);
        assert_eq!(error_count, 0, "Error count should be 0: {}", error_count);
    }

    #[test]
    fn test_concurrent_access_simulation_with_error_count() {
        const NUM_THREADS: usize = 10;
        const NUM_RECORDS: usize = 100;

        const SLEEP_INTERVAL: usize = 10;
        const SLEEP_DURATION_MS: u64 = 1;

        let config = RateLimiterConfig {
            ttft_threshold_secs: 1.0,
            itl_threshold_secs: 0.1,
            time_constant_secs: 30.0,
            per_model_limits: false,
            is_enabled: true,
        };
        let limiter = Arc::new(RateLimiter::new(config));
        let error_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();

        for i in 0..NUM_THREADS {
            let limiter_clone = limiter.clone();
            let error_count_clone = error_count.clone();

            handles.push(thread::spawn(move || {
                for j in 0..NUM_RECORDS {
                    limiter_clone.record_ttft("model", (i * NUM_RECORDS + j) as f64 / 1_000.0);
                    limiter_clone.record_itl("model", (i + j) as f64 / 100.0);

                    if limiter_clone.should_reject("model").should_reject {
                        error_count_clone.fetch_add(1, Ordering::Relaxed);
                    }

                    if j % SLEEP_INTERVAL == 0 {
                        thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Roughly 10% of the time, we should have an error, we set the threshold to 12% to account for
        // the effect of the time passing.
        let error_count = error_count.load(Ordering::Relaxed);
        assert!(
            error_count > 870 && error_count < 930,
            "Error count should be around 12% of the time: {}",
            error_count
        );
    }

    #[test]
    fn test_concurrent_operations() {
        use std::sync::Mutex;

        const TIME_CONSTANT_SECS: f64 = 10.0;

        const SLEEP_DURATION_MS: u64 = 1;

        const NUM_THREADS: usize = 5;
        const NUM_RECORDS: usize = 20;

        let tracker = Arc::new(Mutex::new(TimeWeightedAverageTracker::new(
            TIME_CONSTANT_SECS,
        )));

        let mut handles = Vec::new();

        // Spawn multiple threads adding values
        for thread_id in 0..NUM_THREADS {
            let tracker_clone = Arc::clone(&tracker);
            let handle = thread::spawn(move || {
                for i in 0..NUM_RECORDS {
                    let value = (thread_id * 100 + i) as f64;
                    tracker_clone.lock().unwrap().record_value(value);
                    thread::sleep(Duration::from_millis(SLEEP_DURATION_MS));
                }
            });
            handles.push(handle);
        }

        // Also spawn a thread that computes averages
        const NUM_AVERAGES: usize = 10;
        const SLEEP_DURATION_MS_AVERAGE: u64 = 5;

        let tracker_clone = Arc::clone(&tracker);
        let avg_handle = thread::spawn(move || {
            for _ in 0..NUM_AVERAGES {
                let avg = tracker_clone
                    .lock()
                    .unwrap()
                    .get_decayed_time_weighted_average();
                assert!(avg > 0.0, "Average should be positive");
                thread::sleep(Duration::from_millis(SLEEP_DURATION_MS_AVERAGE));
            }
        });

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        avg_handle.join().unwrap();

        let final_avg = tracker.lock().unwrap().get_decayed_time_weighted_average();
        assert!(final_avg > 0.0, "Final average should be positive");
    }

    #[test]
    fn test_rate_limiter_integration() {
        let config = RateLimiterConfig {
            ttft_threshold_secs: 100., // 100ms
            itl_threshold_secs: 1.,    // 5ms
            time_constant_secs: 1.0,
            ..Default::default()
        };

        let limiter = Arc::new(RateLimiter::new(config));

        // Record low values - should not trigger
        limiter.record_ttft("test", 50.0);
        limiter.record_ttft("test", 60.0);
        limiter.record_ttft("test", 70.0);

        thread::sleep(Duration::from_millis(150)); // Wait for warmup

        assert!(
            !limiter.should_reject("test").should_reject,
            "Should not reject with low values"
        );

        // Record high values - should trigger
        limiter.record_ttft("test", 200.0);
        limiter.record_ttft("test", 300.0);

        assert!(
            limiter.should_reject("test").should_reject,
            "Should reject with high values"
        );
    }

    #[test]
    fn test_rate_limiter_integration_samples_close_to_trigger() {
        const NUM_SAMPLES: usize = 100;

        let config = RateLimiterConfig {
            ttft_threshold_secs: 70.,
            itl_threshold_secs: 0.005,
            time_constant_secs: 1.0,
            ..Default::default()
        };

        let limiter = Arc::new(RateLimiter::new(config));

        // Record low values - should not trigger
        limiter.record_ttft("test", 50.0);
        limiter.record_ttft("test", 60.0);
        limiter.record_ttft("test", 70.0);

        thread::sleep(Duration::from_millis(150)); // Wait for warmup

        assert!(
            !limiter.should_reject("test").should_reject,
            "Should not reject with low values"
        );

        // Record multiple values close to trigger
        for i in 0..NUM_SAMPLES {
            limiter.record_ttft("test", 100.0 + i as f64 / 10.0);
        }

        assert!(
            limiter.should_reject("test").should_reject,
            "Should reject with high values"
        );
    }

    #[test]
    fn test_per_model_vs_global_limits() {
        const MODEL_A: &str = "model_a";
        const MODEL_B: &str = "model_b";

        let global_config = RateLimiterConfig {
            per_model_limits: false,
            ..Default::default()
        };

        let per_model_config = RateLimiterConfig {
            per_model_limits: true,
            ..Default::default()
        };

        let global_limiter = RateLimiter::new(global_config);
        let per_model_limiter = RateLimiter::new(per_model_config);

        // Record high values for model A
        global_limiter.record_ttft(MODEL_A, 2000.0);
        global_limiter.record_ttft(MODEL_A, 2000.0);

        per_model_limiter.record_ttft(MODEL_A, 2000.0);
        per_model_limiter.record_ttft(MODEL_A, 2000.0);

        thread::sleep(Duration::from_millis(20));

        // Both should reject model A
        assert!(global_limiter.should_reject(MODEL_A).should_reject);
        assert!(per_model_limiter.should_reject(MODEL_A).should_reject);

        // Global limiter should also reject model B (uses same "global" key)
        assert!(global_limiter.should_reject(MODEL_B).should_reject);

        // Per-model limiter should NOT reject model B (separate tracking)
        assert!(!per_model_limiter.should_reject(MODEL_B).should_reject);
    }

    #[test]
    fn test_numerical_stability_long_time_series() {
        // Scenario 1: Very small time constant with long time series
        {
            const TIME_CONSTANT_SECS: f64 = 0.01; // Very small time constant
            const NUM_SAMPLES: usize = 10_000;
            const SLEEP_DURATION_MICROS: u64 = 100; // 0.1ms between samples

            let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

            // Add many samples with controlled timing
            for i in 0..NUM_SAMPLES {
                tracker.record_value((i % 100) as f64); // Cycling values 0-99

                if i % 1000 == 0 {
                    // Add occasional small delays to test very old samples
                    thread::sleep(Duration::from_micros(SLEEP_DURATION_MICROS));
                }
            }

            let avg = tracker.get_decayed_time_weighted_average();

            // Should be finite and reasonable
            assert!(
                avg.is_finite(),
                "Average should be finite with small time constant"
            );
            assert!(
                avg > 0.0 && avg < 100.0,
                "Average should be bounded by sample range: {}",
                avg
            );

            // With small time constant, should be dominated by recent samples (90-99 range)
            assert!(
                (avg - 50.0).abs() < 0.5,
                "With small time constant, average should reflect recent samples: {}",
                avg
            );
        }

        // Scenario 2: Extreme value ranges
        {
            const TIME_CONSTANT_SECS: f64 = 1.0;
            let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

            let extreme_values = vec![
                1e-10,     // Very small positive
                1e10,      // Very large
                1e-15,     // Tiny
                1e15,      // Huge
                0.001,     // Small
                1000000.0, // Large
            ];

            for &value in &extreme_values {
                tracker.record_value(value);
                thread::sleep(Duration::from_millis(1));
            }

            let avg = tracker.get_decayed_time_weighted_average();
            assert!(
                avg.is_finite(),
                "Average should handle extreme values gracefully"
            );
            assert!(avg > 0.0, "Average of positive values should be positive");
        }

        // Scenario 3: Accumulated precision test with repetitive operations
        {
            const TIME_CONSTANT_SECS: f64 = 10.0;
            const NUM_ITERATIONS: usize = 50_000;

            let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

            // Record the same value many times to test for accumulated rounding errors
            let test_value = 42.42424242424242; // Value with many decimal places

            for _ in 0..NUM_ITERATIONS {
                tracker.record_value(test_value);
            }

            let avg = tracker.get_decayed_time_weighted_average();

            // Should be very close to the test value (within reasonable floating point precision)
            let relative_error = (avg - test_value).abs() / test_value;
            assert!(
                relative_error < 1e-6,
                "Accumulated rounding error too large: {} vs {}, error: {:.2e}",
                avg,
                test_value,
                relative_error
            );
        }

        // Scenario 4: Weight underflow protection
        {
            const TIME_CONSTANT_SECS: f64 = 0.1; // Small time constant
            let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

            // Add initial sample
            tracker.record_value(1000.0);

            // Wait a very long time (relative to time constant)
            thread::sleep(Duration::from_millis(2000)); // 20 time constants

            // Add recent samples - old sample should have negligible weight
            for i in 0..100 {
                tracker.record_value(100.0 + i as f64);
                thread::sleep(Duration::from_micros(100));
            }

            let avg = tracker.get_decayed_time_weighted_average();

            // Should be dominated by recent samples, not the old high value
            assert!(
                avg < 500.0,
                "Very old samples should have negligible impact: {}",
                avg
            );
            assert!(avg > 100.0, "Average should still be reasonable: {}", avg);
        }

        // Scenario 5: Monotonic behavior verification
        {
            const TIME_CONSTANT_SECS: f64 = 5.0;
            let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

            // Add samples in strictly increasing order
            let mut previous_avg = 0.0;
            for i in 1..=1000 {
                tracker.record_value(i as f64);

                if i % 100 == 0 {
                    let current_avg = tracker.get_decayed_time_weighted_average();

                    // Average should generally increase when adding larger values
                    assert!(
                        current_avg > previous_avg,
                        "Average should increase with larger values: {} -> {} at iteration {}",
                        previous_avg,
                        current_avg,
                        i
                    );

                    previous_avg = current_avg;
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }

        // Scenario 6: Stability under rapid updates
        {
            const TIME_CONSTANT_SECS: f64 = 2.0;
            let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

            // Rapidly add many samples without any delays
            for i in 0..100_000 {
                tracker.record_value((i % 10) as f64); // Values 0-9
            }

            let avg = tracker.get_decayed_time_weighted_average();

            assert!(avg.is_finite(), "Rapid updates should maintain stability");
            assert!(
                (0.0..=9.0).contains(&avg),
                "Average should be bounded: {}",
                avg
            );

            // Should be close to the mean of 0-9 = 4.5
            assert!(
                (avg - 4.5).abs() < 1.0,
                "Average should be close to sample mean with rapid updates: {}",
                avg
            );
        }

        // Scenario 7: Verify internal weight tracking remains stable
        {
            const TIME_CONSTANT_SECS: f64 = 1.0;
            let mut tracker = TimeWeightedAverageTracker::new(TIME_CONSTANT_SECS);

            // Add samples over a long period to test weight accumulation
            for i in 0..1000 {
                tracker.record_value(50.0); // Constant value

                if i % 100 == 0 {
                    thread::sleep(Duration::from_millis(100));
                }
            }

            // Internal weights should be reasonable (not infinite or zero)
            let avg = tracker.get_decayed_time_weighted_average();
            assert!(avg.is_finite(), "Internal state should remain stable");
            assert!(
                (avg - 50.0).abs() < 1e-4,
                "Constant values should maintain constant average"
            );

            // Test that the tracker can still respond to new values
            tracker.record_value(100.0);
            let new_avg = tracker.get_decayed_time_weighted_average();
            assert!(
                new_avg > 50.0,
                "Tracker should still respond to new values after long series"
            );
        }

        println!("✓ All numerical stability tests passed");
    }
}
