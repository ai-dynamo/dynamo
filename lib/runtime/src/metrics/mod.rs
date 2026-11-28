// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics of the runtime.
//!
//! The registry, the metric names, the typed helpers and the frontend pipeline instruments live
//! in `dynamo-metrics` and are re-exported here by name. The request metrics and the load frames
//! are not: take them from `dynamo_metrics`. The environment names are reachable through
//! `crate::config::environment_names::llm::metrics`, which takes its values from that crate.
//! This module adds what needs the runtime: the transport and Tokio instruments and the OTLP
//! exporter.

pub use dynamo_metrics::{
    AttachedPrometheusRegistry, CollectedFamilies, MetricOptions, Metrics, MetricsHierarchy,
    MetricsRegistry, PrometheusExpositionFormatCallback, PrometheusMetric, PrometheusTypedCallback,
    PrometheusUpdateCallback, create_metric, create_metric_with_options, frontend_perf, prom_typed,
    prometheus_names,
};

pub mod otlp_export;
pub mod quic_response;
pub mod request_plane;
pub mod tokio_perf;
pub mod transport_metrics;
pub mod work_handler_perf;
pub mod work_handler_pool;

#[cfg(test)]
mod tests;
