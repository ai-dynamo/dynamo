// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-plane metrics for AddressedPushRouter.
//! Used to pinpoint serialization vs transport roundtrip latency.

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Gauge, Histogram, HistogramOpts};

use super::prometheus_names::{name_prefix, request_plane};
use crate::MetricsRegistry;

fn request_plane_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::REQUEST_PLANE, suffix)
}

/// Time from generate() entry to send_request() (serialization + encoding + control message).
pub static REQUEST_PLANE_QUEUE_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            request_plane_metric_name(request_plane::QUEUE_SECONDS),
            "Time from generate() entry to send_request() (seconds)",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
        ]),
    )
    .expect("request_plane_queue_seconds histogram")
});

/// Time for send_request() to complete (frontend view: network + queue + ack).
pub static REQUEST_PLANE_SEND_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            request_plane_metric_name(request_plane::SEND_SECONDS),
            "Time for send_request() to complete (seconds)",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
        ]),
    )
    .expect("request_plane_send_seconds histogram")
});

/// Time from send_request() to first response item (transport roundtrip TTFT).
pub static REQUEST_PLANE_ROUNDTRIP_TTFT_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            request_plane_metric_name(request_plane::ROUNDTRIP_TTFT_SECONDS),
            "Time from send_request() to first response item (seconds)",
        )
        .buckets(vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
        ]),
    )
    .expect("request_plane_roundtrip_ttft_seconds histogram")
});

/// Currently in-flight requests (incremented at generate() entry, decremented on stream complete).
pub static REQUEST_PLANE_INFLIGHT: Lazy<Gauge> = Lazy::new(|| {
    Gauge::new(
        request_plane_metric_name(request_plane::INFLIGHT_REQUESTS),
        "Currently in-flight requests at AddressedPushRouter",
    )
    .expect("request_plane_inflight gauge")
});

/// Guards idempotency for the `MetricsRegistry` registration path.
static METRICS_REGISTERED: OnceCell<()> = OnceCell::new();

/// Guards idempotency for the raw `prometheus::Registry` registration path.
/// Kept separate from `METRICS_REGISTERED` so that calling `ensure_request_plane_metrics_registered`
/// first does not silently prevent the metrics from being registered in the prometheus registry.
static PROMETHEUS_REGISTERED: OnceCell<Result<(), String>> = OnceCell::new();

/// Register request-plane metrics with the given registry. Idempotent; only the first call registers.
pub fn ensure_request_plane_metrics_registered(registry: &MetricsRegistry) {
    let _ = METRICS_REGISTERED.get_or_init(|| {
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_QUEUE_SECONDS.clone()),
            "request_plane_queue_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_SEND_SECONDS.clone()),
            "request_plane_send_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_ROUNDTRIP_TTFT_SECONDS.clone()),
            "request_plane_roundtrip_ttft_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_INFLIGHT.clone()),
            "request_plane_inflight",
        );
    });
}

/// Register request-plane metrics with a raw Prometheus registry (e.g. for LLM HTTP service /metrics).
/// Idempotent; only the first call registers. Call this when the service exposes /metrics from its own registry.
pub fn ensure_request_plane_metrics_registered_prometheus(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    PROMETHEUS_REGISTERED
        .get_or_init(|| {
            (|| -> Result<(), prometheus::Error> {
                registry.register(Box::new(REQUEST_PLANE_QUEUE_SECONDS.clone()))?;
                registry.register(Box::new(REQUEST_PLANE_SEND_SECONDS.clone()))?;
                registry.register(Box::new(REQUEST_PLANE_ROUNDTRIP_TTFT_SECONDS.clone()))?;
                registry.register(Box::new(REQUEST_PLANE_INFLIGHT.clone()))?;
                Ok(())
            })()
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(|_| ())
        .map_err(|e| prometheus::Error::Msg(e.clone()))
}
