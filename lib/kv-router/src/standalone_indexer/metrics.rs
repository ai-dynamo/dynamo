// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "metrics")]
use std::sync::LazyLock;
#[cfg(feature = "metrics")]
use std::time::Instant;

#[cfg(feature = "metrics")]
use axum::{extract::MatchedPath, http::Request, middleware::Next, response::Response};
#[cfg(feature = "metrics")]
use dynamo_runtime::metrics::prometheus_names::{kvindexer, name_prefix};
#[cfg(feature = "metrics")]
use prometheus::{
    HistogramVec, IntCounterVec, IntGauge, Opts, exponential_buckets, histogram_opts,
};

#[cfg(feature = "metrics")]
pub struct StandaloneIndexerMetrics {
    pub request_duration: HistogramVec,
    pub requests_total: IntCounterVec,
    pub errors_total: IntCounterVec,
    pub models: IntGauge,
    pub workers: IntGauge,
}

#[cfg(feature = "metrics")]
static METRICS: LazyLock<StandaloneIndexerMetrics> = LazyLock::new(|| {
    let prefix = name_prefix::KVINDEXER;
    StandaloneIndexerMetrics {
        request_duration: HistogramVec::new(
            histogram_opts!(
                format!("{prefix}_{}", kvindexer::REQUEST_DURATION_SECONDS),
                "HTTP request latency",
                exponential_buckets(0.0001, 2.0, 20).expect("valid bucket params")
            ),
            &["endpoint"],
        )
        .expect("valid histogram"),
        requests_total: IntCounterVec::new(
            Opts::new(
                format!("{prefix}_{}", kvindexer::REQUESTS_TOTAL),
                "Total HTTP requests",
            ),
            &["endpoint", "method"],
        )
        .expect("valid counter"),
        errors_total: IntCounterVec::new(
            Opts::new(
                format!("{prefix}_{}", kvindexer::ERRORS_TOTAL),
                "HTTP error responses (4xx/5xx)",
            ),
            &["endpoint", "status_class"],
        )
        .expect("valid counter"),
        models: IntGauge::new(
            format!("{prefix}_{}", kvindexer::MODELS),
            "Number of active model+tenant indexers",
        )
        .expect("valid gauge"),
        workers: IntGauge::new(
            format!("{prefix}_{}", kvindexer::WORKERS),
            "Number of registered worker instances",
        )
        .expect("valid gauge"),
    }
});

#[cfg(feature = "metrics")]
pub fn register(registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
    let m = &*METRICS;
    registry.register(Box::new(m.request_duration.clone()))?;
    registry.register(Box::new(m.requests_total.clone()))?;
    registry.register(Box::new(m.errors_total.clone()))?;
    registry.register(Box::new(m.models.clone()))?;
    registry.register(Box::new(m.workers.clone()))?;
    Ok(())
}

#[cfg(feature = "metrics")]
pub async fn metrics_middleware(req: Request<axum::body::Body>, next: Next) -> Response {
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let method = req.method().as_str().to_owned();
    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();
    let m = &*METRICS;
    m.requests_total
        .with_label_values(&[path.as_str(), method.as_str()])
        .inc();
    m.request_duration
        .with_label_values(&[path.as_str()])
        .observe(elapsed);
    let status = response.status().as_u16();
    if status >= 400 {
        let class = if status < 500 { "4xx" } else { "5xx" };
        m.errors_total
            .with_label_values(&[path.as_str(), class])
            .inc();
    }
    response
}

#[cfg(feature = "metrics")]
pub fn inc_models() {
    METRICS.models.inc();
}

#[cfg(not(feature = "metrics"))]
pub fn inc_models() {}

#[cfg(feature = "metrics")]
pub fn inc_workers() {
    METRICS.workers.inc();
}

#[cfg(not(feature = "metrics"))]
pub fn inc_workers() {}

#[cfg(feature = "metrics")]
pub fn dec_workers() {
    METRICS.workers.dec();
}

#[cfg(not(feature = "metrics"))]
pub fn dec_workers() {}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use super::*;
    use prometheus::Encoder;

    #[test]
    fn register_and_encode() {
        let registry = prometheus::Registry::new();
        register(&registry).expect("registration should succeed");

        inc_models();
        inc_workers();
        inc_workers();
        dec_workers();

        let encoder = prometheus::TextEncoder::new();
        let mut buf = Vec::new();
        encoder.encode(&registry.gather(), &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("dynamo_kvindexer_request_duration_seconds"));
        assert!(output.contains("dynamo_kvindexer_requests_total"));
        assert!(output.contains("dynamo_kvindexer_errors_total"));
        assert!(output.contains("dynamo_kvindexer_models 1"));
        assert!(output.contains("dynamo_kvindexer_workers 1"));
    }
}
