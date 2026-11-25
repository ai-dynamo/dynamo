// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metrics of the runtime.
//!
//! The registry, the metric names and the typed helpers live in `dynamo-metrics` and are
//! re-exported here. This module adds what needs the runtime: the transport and Tokio
//! instruments and the OTLP exporter.

pub use dynamo_metrics::*;

pub mod otlp_export;
pub mod quic_response;
pub mod request_plane;
pub mod tokio_perf;
pub mod transport_metrics;
pub mod work_handler_perf;
pub mod work_handler_pool;

#[cfg(test)]
mod tests;
