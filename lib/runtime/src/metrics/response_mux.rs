// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded-cardinality metrics for the multiplexed TCP response transport.

use std::sync::{Mutex, RwLock, Weak};

use once_cell::sync::Lazy;
use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGaugeVec, Opts,
};

use crate::MetricsRegistry;

pub static ACTIVE_CONNECTIONS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_active_connections",
            "Active physical TCP response-mux connections",
        ),
        &["role"],
    )
    .expect("response mux active connection gauge")
});

pub static CONNECTIONS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_connections_total",
            "Physical TCP response-mux connection lifecycle events",
        ),
        &["role", "result"],
    )
    .expect("response mux connection counter")
});

pub static ACTIVE_STREAMS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_active_streams",
            "Active logical response streams",
        ),
        &["role"],
    )
    .expect("response mux active stream gauge")
});

pub static SETUP_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_setup_seconds",
            "Time from request dispatch to logical response-stream prologue",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
        ]),
    )
    .expect("response mux setup histogram")
});

pub static FRAMES_PER_WRITE: Lazy<HistogramVec> = Lazy::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_frames_per_write",
            "Logical response-mux frames encoded into each physical write",
        )
        .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
        &["role"],
    )
    .expect("response mux frames-per-write histogram")
});

pub static CONFIGURED_BATCH_INTERVAL_MS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_configured_batch_interval_ms",
            "Configured response data batching interval in milliseconds",
        ),
        &["role"],
    )
    .expect("response mux configured batch interval gauge")
});

pub static WRITE_CALLS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_write_calls_total",
            "Physical response-mux TCP write calls",
        ),
        &["role"],
    )
    .expect("response mux write call counter")
});

pub static DATA_SEGMENTS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_data_segments_total",
            "Kernel TCP data segments sent on response sockets when packet metrics are enabled",
        ),
        &["transport", "role"],
    )
    .expect("response TCP data segment counter")
});

pub static RESETS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_resets_total",
            "Logical response streams reset by role and reason",
        ),
        &["role", "reason"],
    )
    .expect("response mux reset counter")
});

pub static RECONNECTS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_reconnects_total",
            "Replacement physical response-mux connections",
        ),
        &["role"],
    )
    .expect("response mux reconnect counter")
});

pub static STALLS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_stalls_total",
            "Response producer stalls by bounded admission point",
        ),
        &["kind"],
    )
    .expect("response mux stall counter")
});

pub static CONNECTION_LOST_STREAMS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    IntCounter::new(
        "dynamo_tcp_response_mux_connection_lost_streams_total",
        "Logical streams failed by physical response-mux connection loss",
    )
    .expect("response mux connection-lost stream counter")
});

static REGISTERED: Lazy<Mutex<Vec<Weak<RwLock<prometheus::Registry>>>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

pub fn ensure_registered(registry: &MetricsRegistry) {
    {
        let mut registered = REGISTERED.lock().expect("response mux registry lock");
        registered.retain(|candidate| candidate.strong_count() > 0);
        let identity = std::sync::Arc::downgrade(&registry.prometheus_registry);
        if registered
            .iter()
            .any(|candidate| Weak::ptr_eq(candidate, &identity))
        {
            return;
        }
        registered.push(identity);
    }

    registry.add_metric_or_warn(
        Box::new(ACTIVE_CONNECTIONS.clone()),
        "response_mux_active_connections",
    );
    registry.add_metric_or_warn(
        Box::new(CONNECTIONS_TOTAL.clone()),
        "response_mux_connections_total",
    );
    registry.add_metric_or_warn(
        Box::new(ACTIVE_STREAMS.clone()),
        "response_mux_active_streams",
    );
    registry.add_metric_or_warn(
        Box::new(SETUP_SECONDS.clone()),
        "response_mux_setup_seconds",
    );
    registry.add_metric_or_warn(
        Box::new(FRAMES_PER_WRITE.clone()),
        "response_mux_frames_per_write",
    );
    registry.add_metric_or_warn(
        Box::new(CONFIGURED_BATCH_INTERVAL_MS.clone()),
        "response_mux_configured_batch_interval_ms",
    );
    registry.add_metric_or_warn(
        Box::new(WRITE_CALLS_TOTAL.clone()),
        "response_mux_write_calls_total",
    );
    registry.add_metric_or_warn(
        Box::new(DATA_SEGMENTS_TOTAL.clone()),
        "response_data_segments_total",
    );
    registry.add_metric_or_warn(Box::new(RESETS_TOTAL.clone()), "response_mux_resets_total");
    registry.add_metric_or_warn(
        Box::new(RECONNECTS_TOTAL.clone()),
        "response_mux_reconnects_total",
    );
    registry.add_metric_or_warn(Box::new(STALLS_TOTAL.clone()), "response_mux_stalls_total");
    registry.add_metric_or_warn(
        Box::new(CONNECTION_LOST_STREAMS_TOTAL.clone()),
        "response_mux_connection_lost_streams_total",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_register_with_multiple_registries() {
        let first = MetricsRegistry::new();
        let second = MetricsRegistry::new();
        ensure_registered(&first);
        ensure_registered(&first);
        ensure_registered(&second);
    }
}
