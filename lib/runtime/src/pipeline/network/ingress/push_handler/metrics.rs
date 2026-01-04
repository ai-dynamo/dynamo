// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics for work handler profiling and request tracking.

use crate::metrics::MetricsHierarchy;
use crate::metrics::prometheus_names::work_handler;
use prometheus::{Histogram, IntCounter, IntCounterVec, IntGauge};
use std::time::Instant;

/// Metrics configuration for profiling work handlers
#[derive(Clone, Debug)]
pub struct WorkHandlerMetrics {
    pub request_counter: IntCounter,
    pub request_duration: Histogram,
    pub inflight_requests: IntGauge,
    pub request_bytes: IntCounter,
    pub response_bytes: IntCounter,
    pub error_counter: IntCounterVec,
}

impl WorkHandlerMetrics {
    pub fn new(
        request_counter: IntCounter,
        request_duration: Histogram,
        inflight_requests: IntGauge,
        request_bytes: IntCounter,
        response_bytes: IntCounter,
        error_counter: IntCounterVec,
    ) -> Self {
        Self {
            request_counter,
            request_duration,
            inflight_requests,
            request_bytes,
            response_bytes,
            error_counter,
        }
    }

    /// Create WorkHandlerMetrics from an endpoint using its built-in labeling
    pub fn from_endpoint(
        endpoint: &crate::component::Endpoint,
        metrics_labels: Option<&[(&str, &str)]>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let metrics_labels = metrics_labels.unwrap_or(&[]);
        let metrics = endpoint.metrics();
        let request_counter = metrics.create_intcounter(
            work_handler::REQUESTS_TOTAL,
            "Total number of requests processed by work handler",
            metrics_labels,
        )?;

        let request_duration = metrics.create_histogram(
            work_handler::REQUEST_DURATION_SECONDS,
            "Time spent processing requests by work handler",
            metrics_labels,
            None,
        )?;

        let inflight_requests = metrics.create_intgauge(
            work_handler::INFLIGHT_REQUESTS,
            "Number of requests currently being processed by work handler",
            metrics_labels,
        )?;

        let request_bytes = metrics.create_intcounter(
            work_handler::REQUEST_BYTES_TOTAL,
            "Total number of bytes received in requests by work handler",
            metrics_labels,
        )?;

        let response_bytes = metrics.create_intcounter(
            work_handler::RESPONSE_BYTES_TOTAL,
            "Total number of bytes sent in responses by work handler",
            metrics_labels,
        )?;

        let error_counter = metrics.create_intcountervec(
            work_handler::ERRORS_TOTAL,
            "Total number of errors in work handler processing",
            &[work_handler::ERROR_TYPE_LABEL],
            metrics_labels,
        )?;

        Ok(Self::new(
            request_counter,
            request_duration,
            inflight_requests,
            request_bytes,
            response_bytes,
            error_counter,
        ))
    }
}

/// RAII guard to ensure inflight gauge is decremented and request duration is observed on all code paths.
pub(super) struct RequestMetricsGuard {
    inflight_requests: IntGauge,
    request_duration: Histogram,
    start_time: Instant,
}

impl RequestMetricsGuard {
    /// Create a new metrics guard from existing metric handles.
    pub fn new(metrics: &WorkHandlerMetrics, start_time: Instant) -> Self {
        Self {
            inflight_requests: metrics.inflight_requests.clone(),
            request_duration: metrics.request_duration.clone(),
            start_time,
        }
    }
}

impl Drop for RequestMetricsGuard {
    fn drop(&mut self) {
        self.inflight_requests.dec();
        self.request_duration
            .observe(self.start_time.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::{HistogramOpts, Opts};

    /// Create standalone test metrics (not tied to an Endpoint)
    fn create_test_metrics() -> WorkHandlerMetrics {
        WorkHandlerMetrics {
            request_counter: IntCounter::new("test_requests", "test").unwrap(),
            request_duration: Histogram::with_opts(HistogramOpts::new("test_duration", "test"))
                .unwrap(),
            inflight_requests: IntGauge::new("test_inflight", "test").unwrap(),
            request_bytes: IntCounter::new("test_req_bytes", "test").unwrap(),
            response_bytes: IntCounter::new("test_resp_bytes", "test").unwrap(),
            error_counter: IntCounterVec::new(Opts::new("test_errors", "test"), &["error_type"])
                .unwrap(),
        }
    }

    #[test]
    fn test_request_metrics_guard_decrements_inflight_on_drop() {
        let metrics = create_test_metrics();
        metrics.inflight_requests.inc(); // Simulate request start
        assert_eq!(metrics.inflight_requests.get(), 1);

        {
            let _guard = RequestMetricsGuard::new(&metrics, Instant::now());
            assert_eq!(metrics.inflight_requests.get(), 1);
        } // guard dropped here

        assert_eq!(metrics.inflight_requests.get(), 0);
    }

    #[test]
    fn test_request_metrics_guard_records_duration_on_drop() {
        let metrics = create_test_metrics();

        {
            let start = Instant::now();
            std::thread::sleep(std::time::Duration::from_millis(10));
            let _guard = RequestMetricsGuard::new(&metrics, start);
        }

        // Histogram should have 1 observation
        assert_eq!(metrics.request_duration.get_sample_count(), 1);
        // Duration should be >= 10ms
        assert!(metrics.request_duration.get_sample_sum() >= 0.01);
    }

    #[test]
    fn test_guard_handles_panic_path() {
        let metrics = create_test_metrics();
        metrics.inflight_requests.inc();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = RequestMetricsGuard::new(&metrics, Instant::now());
            panic!("simulated panic");
        }));

        assert!(result.is_err());
        // Guard should still have decremented on unwind
        assert_eq!(metrics.inflight_requests.get(), 0);
    }

    #[test]
    fn test_new_assigns_all_fields() {
        let metrics = create_test_metrics();
        // Verify fields are accessible and work correctly
        metrics.request_counter.inc();
        assert_eq!(metrics.request_counter.get(), 1);

        metrics.request_bytes.inc_by(100);
        assert_eq!(metrics.request_bytes.get(), 100);

        metrics.response_bytes.inc_by(200);
        assert_eq!(metrics.response_bytes.get(), 200);

        metrics.inflight_requests.inc();
        metrics.inflight_requests.inc();
        assert_eq!(metrics.inflight_requests.get(), 2);

        metrics
            .error_counter
            .with_label_values(&["test_error"])
            .inc();
        assert_eq!(
            metrics
                .error_counter
                .with_label_values(&["test_error"])
                .get(),
            1
        );
    }
}
