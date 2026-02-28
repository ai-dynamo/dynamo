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
        request_plane_metric_name(request_plane::INFLIGHT),
        "Currently in-flight requests at AddressedPushRouter",
    )
    .expect("request_plane_inflight gauge")
});

static REGISTERED: OnceCell<()> = OnceCell::new();

/// Register request-plane metrics with the given registry. Idempotent; only the first call registers.
pub fn ensure_request_plane_metrics_registered(registry: &MetricsRegistry) {
    let _ = REGISTERED.get_or_init(|| {
        registry
            .add_metric(Box::new(REQUEST_PLANE_QUEUE_SECONDS.clone()))
            .ok();
        registry
            .add_metric(Box::new(REQUEST_PLANE_SEND_SECONDS.clone()))
            .ok();
        registry
            .add_metric(Box::new(REQUEST_PLANE_ROUNDTRIP_TTFT_SECONDS.clone()))
            .ok();
        registry
            .add_metric(Box::new(REQUEST_PLANE_INFLIGHT.clone()))
            .ok();
    });
}

/// Register request-plane metrics with a raw Prometheus registry (e.g. for LLM HTTP service /metrics).
/// Idempotent; only the first call registers. Call this when the service exposes /metrics from its own registry.
pub fn ensure_request_plane_metrics_registered_prometheus(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    if REGISTERED.get().is_some() {
        return Ok(());
    }
    registry.register(Box::new(REQUEST_PLANE_QUEUE_SECONDS.clone()))?;
    registry.register(Box::new(REQUEST_PLANE_SEND_SECONDS.clone()))?;
    registry.register(Box::new(REQUEST_PLANE_ROUNDTRIP_TTFT_SECONDS.clone()))?;
    registry.register(Box::new(REQUEST_PLANE_INFLIGHT.clone()))?;
    let _ = REGISTERED.set(());
    Ok(())
}
