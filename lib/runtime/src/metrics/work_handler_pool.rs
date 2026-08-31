// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-pool saturation metrics for the shared TCP server (backend side).
//!
//! These metrics expose queue buildup between `work_tx.send()` and dispatcher
//! pickup, and permit starvation in the bounded worker pool.

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge};

use super::prometheus_names::{name_prefix, work_handler};
use crate::MetricsRegistry;

fn work_handler_metric_name(suffix: &str) -> String {
    format!("{}_{}", name_prefix::WORK_HANDLER, suffix)
}

/// Requests shed because the worker was at capacity: all engine in-flight slots
/// (`--engine-request-limit`) held AND the overflow queue
/// (`--dynamo-request-queue-limit`) full.
pub static REJECTION_REQUEST_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    IntCounter::new(
        format!("{}_request_total", name_prefix::REJECTION),
        "Requests rejected because the worker is at capacity (engine in-flight limit and Dynamo queue both full)",
    )
    .expect("rejection_request_total counter")
});

/// Requests currently in the engine (worker-pool permits in use). Driven from
/// `ActiveTaskGuard`, alongside `WORK_HANDLER_POOL_ACTIVE_TASKS`.
pub static ENGINE_REQUEST_GAUGE: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        "dynamo_engine_request",
        "Current number of requests being handled by the engine (--engine-request-limit)",
    )
    .expect("dynamo_engine_request gauge")
});

/// Requests queued in Dynamo but not yet in the engine. Driven alongside
/// `WORK_HANDLER_QUEUE_DEPTH` (inc on enqueue, dec on dispatcher recv).
pub static REQUEST_QUEUE_GAUGE: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        "dynamo_request_queue",
        "Current number of requests queued in Dynamo not yet in the engine (--dynamo-request-queue-limit)",
    )
    .expect("dynamo_request_queue gauge")
});

/// Current items sitting in the bounded mpsc work queue awaiting dispatcher
/// pickup. Incremented on successful `work_tx.send()` and decremented immediately
/// after `work_rx.recv()`. Permit-acquire wait is NOT counted here — see
/// `WORK_HANDLER_PERMIT_WAIT_SECONDS`.
pub static WORK_HANDLER_QUEUE_DEPTH: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        work_handler_metric_name(work_handler::QUEUE_DEPTH),
        "Current items in the bounded work queue awaiting dispatcher pickup",
    )
    .expect("work_handler_queue_depth gauge")
});

/// Configured capacity of the bounded work queue. Static; set once at server init.
pub static WORK_HANDLER_QUEUE_CAPACITY: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        work_handler_metric_name(work_handler::QUEUE_CAPACITY),
        "Configured capacity of the bounded work queue",
    )
    .expect("work_handler_queue_capacity gauge")
});

/// Times enqueuing work failed because the dispatcher channel was closed
/// (receiver gone). A full queue is a rejection, counted by
/// `REJECTION_REQUEST_TOTAL`, not here.
pub static WORK_HANDLER_ENQUEUE_REJECTED_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    IntCounter::new(
        work_handler_metric_name(work_handler::ENQUEUE_REJECTED_TOTAL),
        "Times enqueuing work failed because the dispatcher channel was closed",
    )
    .expect("work_handler_enqueue_rejected_total counter")
});

/// Time spent waiting to acquire a worker-pool permit. Normal operation is
/// sub-millisecond; saturation pushes p99 into seconds.
pub static WORK_HANDLER_PERMIT_WAIT_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            work_handler_metric_name(work_handler::PERMIT_WAIT_SECONDS),
            "Time spent waiting for a worker-pool permit (seconds)",
        )
        .buckets(vec![
            0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0,
        ]),
    )
    .expect("work_handler_permit_wait_seconds histogram")
});

/// Current number of active worker-pool tasks (permits in use).
pub static WORK_HANDLER_POOL_ACTIVE_TASKS: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        work_handler_metric_name(work_handler::POOL_ACTIVE_TASKS),
        "Current number of active worker-pool tasks (permits in use)",
    )
    .expect("work_handler_pool_active_tasks gauge")
});

/// Configured worker-pool capacity (total permits). Static; set once at server init.
pub static WORK_HANDLER_POOL_CAPACITY: Lazy<IntGauge> = Lazy::new(|| {
    IntGauge::new(
        work_handler_metric_name(work_handler::POOL_CAPACITY),
        "Configured worker-pool capacity (total permits)",
    )
    .expect("work_handler_pool_capacity gauge")
});

/// Guards idempotency for the `MetricsRegistry` registration path.
static METRICS_REGISTERED: OnceCell<()> = OnceCell::new();

/// Register worker-pool saturation metrics with the given registry. Idempotent.
pub fn ensure_work_handler_pool_metrics_registered(registry: &MetricsRegistry) {
    let _ = METRICS_REGISTERED.get_or_init(|| {
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_QUEUE_DEPTH.clone()),
            "work_handler_queue_depth",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_QUEUE_CAPACITY.clone()),
            "work_handler_queue_capacity",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_ENQUEUE_REJECTED_TOTAL.clone()),
            "work_handler_enqueue_rejected_total",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_PERMIT_WAIT_SECONDS.clone()),
            "work_handler_permit_wait_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_POOL_ACTIVE_TASKS.clone()),
            "work_handler_pool_active_tasks",
        );
        registry.add_metric_or_warn(
            Box::new(WORK_HANDLER_POOL_CAPACITY.clone()),
            "work_handler_pool_capacity",
        );
        registry.add_metric_or_warn(
            Box::new(REJECTION_REQUEST_TOTAL.clone()),
            "rejection_request_total",
        );
        registry.add_metric_or_warn(
            Box::new(ENGINE_REQUEST_GAUGE.clone()),
            "dynamo_engine_request",
        );
        registry.add_metric_or_warn(
            Box::new(REQUEST_QUEUE_GAUGE.clone()),
            "dynamo_request_queue",
        );
    });
}
