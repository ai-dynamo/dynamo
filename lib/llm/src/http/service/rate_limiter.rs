use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use hdrhistogram::Histogram;

const MODEL_METRICS_PRECISION: u8 = 6;

#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    enabled: bool,
    ttft_threshold_ms: f64,
    ttft_percentile: f64,
    itl_threshold_ms: f64,
    itl_percentile: f64,
    window_duration: Duration,
    per_model_limits: bool,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttft_threshold_ms: 1000.0, // 1s
            itl_threshold_ms: 10.0,    // 10ms
            ttft_percentile: 0.95,     // 95th percentile
            itl_percentile: 0.95,      // 95th percentile
            window_duration: Duration::from_secs(5),
            per_model_limits: false,
        }
    }
}

#[derive(Debug)]
pub struct WindowedHistogram {
    current: Histogram<u64>,
    previous: Option<Histogram<u64>>,
    window_start: Instant,
    window_duration: Duration,
    /// Max value in microseconds
    max_value: u64,
    precision: u8,
}

impl WindowedHistogram {
    pub fn new(window_duration: Duration, max_value: u64, precision: u8) -> Result<Self> {
        let histogram = Histogram::<u64>::new_with_bounds(1, max_value, precision)
            .context("Failed to create histogram")?;

        Ok(Self {
            current: histogram,
            previous: None,
            window_start: Instant::now(),
            window_duration,
            max_value,
            precision,
        })
    }

    fn record_value(&mut self, value: f64) -> Result<()> {
        self.maybe_rotate_window()?;

        // Convert to microseconds for better precision
        let value_us = (value * 1000.0).round() as u64;
        let clamped_value = value_us.min(self.max_value).max(1);
        self.current
            .record(clamped_value)
            .context("Failed to record value to histogram")?;

        Ok(())
    }

    fn get_percentile(&self, percentile: f64) -> Result<f64> {
        let sample_count = self.sample_count();

        if sample_count == 0 {
            return Ok(0.0);
        }

        let percentile_us = self.current.value_at_percentile(percentile) as f64;

        // Convert back from microseconds to milliseconds
        Ok(percentile_us / 1000.0)
    }

    fn sample_count(&self) -> u64 {
        self.current.len() + self.previous.as_ref().map_or(0, |h| h.len())
    }

    fn maybe_rotate_window(&mut self) -> Result<()> {
        let now = Instant::now();
        if now.duration_since(self.window_start) > self.window_duration {
            let new_histogram =
                Histogram::<u64>::new_with_bounds(1, self.max_value, self.precision)
                    .context("Failed to create new histogram")?;

            self.previous = Some(std::mem::replace(&mut self.current, new_histogram));
            self.window_start = now;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct ModelMetrics {
    ttft_histogram: WindowedHistogram,
    itl_histogram: WindowedHistogram,
}

impl ModelMetrics {
    fn new(config: &RateLimiterConfig) -> Result<Self> {
        let ttft_histogram = WindowedHistogram::new(
            config.window_duration,
            (config.ttft_threshold_ms * 1000.0).round() as u64, // Convert to microseconds
            MODEL_METRICS_PRECISION,
        )?;
        let itl_histogram = WindowedHistogram::new(
            config.window_duration,
            (config.itl_threshold_ms * 1000.0).round() as u64, // Convert to microseconds
            MODEL_METRICS_PRECISION,
        )?;

        Ok(Self {
            ttft_histogram,
            itl_histogram,
        })
    }
}

pub struct RateLimiter {
    config: RateLimiterConfig,
    // TODO: Can make this a `DashMap` to avoid the need to lock the entire map
    model_metrics: Arc<RwLock<HashMap<String, ModelMetrics>>>,
}

impl RateLimiter {
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            config,
            model_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
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
    pub fn record_ttft(&self, model: &str, ttft_ms: f64) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let model_key = self.get_model_key(model);
        let mut metrics = self.model_metrics.write().unwrap();
        let model_metrics = metrics
            .entry(model_key)
            .or_insert_with(|| ModelMetrics::new(&self.config).unwrap());

        model_metrics
            .ttft_histogram
            .record_value(ttft_ms)
            .context("Failed to record time to first token metric")
    }

    /// Record the inter-token latency metric for a given model
    pub fn record_itl(&self, model: &str, itl_ms: f64) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let model_key = self.get_model_key(model);
        let mut metrics = self.model_metrics.write().unwrap();
        let model_metrics = metrics
            .entry(model_key)
            .or_insert_with(|| ModelMetrics::new(&self.config).unwrap());

        model_metrics
            .itl_histogram
            .record_value(itl_ms)
            .context("Failed to record inter-token latency metric")
    }

    /// Check if the request should be rejected based on the cached metrics
    ///
    /// Returns true if the request should be rejected, false otherwise
    pub fn should_reject(&self, model: &str) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        let model_key = self.get_model_key(model);
        let metrics = self.model_metrics.write().unwrap();

        let Some(model_metrics) = metrics.get(&model_key) else {
            return Ok(false);
        };

        let ttft_percentile_ms = model_metrics
            .ttft_histogram
            .get_percentile(self.config.ttft_percentile)?;
        let itl_percentile_ms = model_metrics
            .itl_histogram
            .get_percentile(self.config.itl_percentile)?;

        let ttft_samples = model_metrics.ttft_histogram.sample_count();
        let itl_samples = model_metrics.itl_histogram.sample_count();

        // Don't reject if we don't have enough samples
        if ttft_samples == 0 || itl_samples == 0 {
            return Ok(false);
        }

        let ttft_exceeded = self.config.ttft_threshold_ms <= ttft_percentile_ms;
        let itl_exceeded = self.config.itl_threshold_ms <= itl_percentile_ms;

        if ttft_exceeded || itl_exceeded {
            tracing::warn!(
                model = model,
                ttft_threshold_ms = self.config.ttft_threshold_ms,
                itl_threshold_ms = self.config.itl_threshold_ms,
                "Rate limit exceeded for model {model}: ttft: {ttft_percentile_ms}ms, itl: {itl_percentile_ms}ms",
                ttft_percentile_ms = ttft_percentile_ms,
                itl_percentile_ms = itl_percentile_ms,
            );
            return Ok(true);
        }

        Ok(false)
    }

    pub fn clear_model_metrics(&self, model: &str) -> Result<()> {
        let model_key = self.get_model_key(model);
        let mut metrics = self.model_metrics.write().unwrap();
        metrics.remove(&model_key);

        Ok(())
    }
}
