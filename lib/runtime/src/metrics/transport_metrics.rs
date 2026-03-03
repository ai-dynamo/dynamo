// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-layer Prometheus metrics (TCP + NATS).
//! Statics are incremented directly in the client send paths.

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Counter, IntCounterVec, Opts};

use super::prometheus_names::{name_prefix, transport};

fn transport_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::TRANSPORT, suffix)
}

// --- TCP counters ---

pub static TCP_BYTES_SENT_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::new(
        transport_metric_name(transport::tcp::BYTES_SENT_TOTAL),
        "Total bytes sent by TCP request client",
    )
    .expect("tcp_bytes_sent_total counter")
});

pub static TCP_BYTES_RECEIVED_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::new(
        transport_metric_name(transport::tcp::BYTES_RECEIVED_TOTAL),
        "Total bytes received by TCP request client",
    )
    .expect("tcp_bytes_received_total counter")
});

pub static TCP_ERRORS_TOTAL: Lazy<Counter> = Lazy::new(|| {
    Counter::new(
        transport_metric_name(transport::tcp::ERRORS_TOTAL),
        "Total TCP request errors (send failure or timeout)",
    )
    .expect("tcp_errors_total counter")
});

// --- NATS counters ---

/// `error_type` label values: "request_failed"
pub static NATS_ERRORS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            transport_metric_name(transport::nats::ERRORS_TOTAL),
            "Total NATS request errors (label: error_type)",
        ),
        &["error_type"],
    )
    .expect("nats_errors_total counter vec")
});

static REGISTERED: OnceCell<()> = OnceCell::new();

/// Register transport metrics with a raw Prometheus registry. Idempotent.
pub fn ensure_transport_metrics_registered_prometheus(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    if REGISTERED.get().is_some() {
        return Ok(());
    }
    registry.register(Box::new(TCP_BYTES_SENT_TOTAL.clone()))?;
    registry.register(Box::new(TCP_BYTES_RECEIVED_TOTAL.clone()))?;
    registry.register(Box::new(TCP_ERRORS_TOTAL.clone()))?;
    registry.register(Box::new(NATS_ERRORS_TOTAL.clone()))?;
    let _ = REGISTERED.set(());
    Ok(())
}
