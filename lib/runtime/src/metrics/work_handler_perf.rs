// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport breakdown metrics for work handler (backend side).
//! Captures network transit (T2-T1) and backend processing time (T3-T2).

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Histogram, HistogramOpts};

use super::prometheus_names::{name_prefix, work_handler};
use crate::MetricsRegistry;

fn work_handler_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::COMPONENT, suffix)
}

/// Network transit: frontend send to backend receive (wall-clock, cross-process).
pub static WORK_HANDLER_NETWORK_TRANSIT_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            work_handler_metric_name(work_handler::NETWORK_TRANSIT_SECONDS),
            "Frontend-to-backend network transit time (cross-process wall-clock, seconds)",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
        ]),
    )
    .expect("work_handler_network_transit_seconds histogram")
});

/// Backend processing: handle_payload entry to first response sent.
pub static WORK_HANDLER_TIME_TO_FIRST_RESPONSE_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            work_handler_metric_name(work_handler::TIME_TO_FIRST_RESPONSE_SECONDS),
            "Backend processing time from handle_payload entry to prologue sent (seconds)",
        )
        .buckets(vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ]),
    )
    .expect("work_handler_time_to_first_response_seconds histogram")
});

static REGISTERED: OnceCell<()> = OnceCell::new();

/// Register work handler transport breakdown metrics with the given registry. Idempotent.
pub fn ensure_work_handler_perf_metrics_registered(registry: &MetricsRegistry) {
    let _ = REGISTERED.get_or_init(|| {
        registry
            .add_metric(Box::new(WORK_HANDLER_NETWORK_TRANSIT_SECONDS.clone()))
            .ok();
        registry
            .add_metric(Box::new(
                WORK_HANDLER_TIME_TO_FIRST_RESPONSE_SECONDS.clone(),
            ))
            .ok();
    });
}

/// Register with a raw Prometheus registry. Idempotent.
pub fn ensure_work_handler_perf_metrics_registered_prometheus(
    registry: &prometheus::Registry,
) -> Result<(), prometheus::Error> {
    if REGISTERED.get().is_some() {
        return Ok(());
    }
    registry.register(Box::new(WORK_HANDLER_NETWORK_TRANSIT_SECONDS.clone()))?;
    registry.register(Box::new(
        WORK_HANDLER_TIME_TO_FIRST_RESPONSE_SECONDS.clone(),
    ))?;
    let _ = REGISTERED.set(());
    Ok(())
}
