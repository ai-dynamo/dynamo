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

/// Ingress-side ACK flush latency (worker SharedTcpEndpoint write_loop): decoded_at ->
/// socket flush complete. Observed for every traced response (DYN_ACK_TRACE=1), not just
/// ones that cross the DYN_ACK_TRACE_WARN_MS log threshold -- gives the full distribution
/// instead of only the tail that happened to get logged.
pub static REQUEST_PLANE_ACK_FLUSH_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            request_plane_metric_name(request_plane::ACK_FLUSH_SECONDS),
            "Ingress ACK flush latency: decoded_at -> socket flush complete (seconds)",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0,
        ]),
    )
    .expect("request_plane_ack_flush_seconds histogram")
});

/// Time to register the request/response stream halves with the response transport
/// (AddressedPushRouter::register_streams).
pub static REQUEST_PLANE_REGISTER_STREAMS_MS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            request_plane_metric_name(request_plane::REGISTER_STREAMS_MS),
            "Time to register request/response stream halves with the response transport (milliseconds)",
        )
        .buckets(vec![
            0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0,
        ]),
    )
    .expect("request_plane_register_streams_ms histogram")
});

/// Time for the tombstone-check `associate_instance` call against the response transport.
pub static REQUEST_PLANE_ASSOCIATE_INSTANCE_MS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            request_plane_metric_name(request_plane::ASSOCIATE_INSTANCE_MS),
            "Time for the associate_instance tombstone check (milliseconds)",
        )
        .buckets(vec![
            0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0,
        ]),
    )
    .expect("request_plane_associate_instance_ms histogram")
});

/// Time to build the request envelope (build_request_envelope: serialization + control
/// message assembly).
pub static REQUEST_PLANE_BUILD_ENVELOPE_MS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            request_plane_metric_name(request_plane::BUILD_ENVELOPE_MS),
            "Time to build the request envelope (milliseconds)",
        )
        .buckets(vec![
            0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0,
        ]),
    )
    .expect("request_plane_build_envelope_ms histogram")
});

/// Time for dispatch_buffer to complete (transport write of the built envelope). Finer
/// granularity sibling of REQUEST_PLANE_SEND_SECONDS at the same call site.
pub static REQUEST_PLANE_DISPATCH_BUFFER_MS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            request_plane_metric_name(request_plane::DISPATCH_BUFFER_MS),
            "Time for dispatch_buffer to complete (milliseconds)",
        )
        .buckets(vec![
            0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0,
        ]),
    )
    .expect("request_plane_dispatch_buffer_ms histogram")
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
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_ACK_FLUSH_SECONDS.clone()),
            "request_plane_ack_flush_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_REGISTER_STREAMS_MS.clone()),
            "request_plane_register_streams_ms",
        );
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_ASSOCIATE_INSTANCE_MS.clone()),
            "request_plane_associate_instance_ms",
        );
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_BUILD_ENVELOPE_MS.clone()),
            "request_plane_build_envelope_ms",
        );
        registry.add_metric_or_warn(
            Box::new(REQUEST_PLANE_DISPATCH_BUFFER_MS.clone()),
            "request_plane_dispatch_buffer_ms",
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
                registry.register(Box::new(REQUEST_PLANE_ACK_FLUSH_SECONDS.clone()))?;
                registry.register(Box::new(REQUEST_PLANE_REGISTER_STREAMS_MS.clone()))?;
                registry.register(Box::new(REQUEST_PLANE_ASSOCIATE_INSTANCE_MS.clone()))?;
                registry.register(Box::new(REQUEST_PLANE_BUILD_ENVELOPE_MS.clone()))?;
                registry.register(Box::new(REQUEST_PLANE_DISPATCH_BUFFER_MS.clone()))?;
                Ok(())
            })()
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(|_| ())
        .map_err(|e| prometheus::Error::Msg(e.clone()))
}
