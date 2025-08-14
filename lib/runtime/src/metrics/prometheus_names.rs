// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metric name constants
//!
//! This module provides centralized Prometheus metric name constants for various components
//! to ensure consistency and avoid duplication across the codebase.

/// Builds a full metric name by prepending the component prefix
pub fn build_metric_name(metric_name: &str) -> String {
    format!("{}{}", name_prefix::COMPONENT, metric_name)
}

/// Metric name prefixes used across the metrics system
pub mod name_prefix {
    /// Prefix for all Prometheus metric names.
    pub const COMPONENT: &str = "dynamo_component_";

    // TODO(keivenc): uncomment below for the frontend
    // pub const FRONTEND: &str = "dynamo_frontend_";
}

/// Automatically inserted Prometheus label names used across the metrics system
pub mod labels {
    /// Label for component identification
    pub const COMPONENT: &str = "dynamo_component";

    /// Label for namespace identification
    pub const NAMESPACE: &str = "dynamo_namespace";

    /// Label for endpoint identification
    pub const ENDPOINT: &str = "dynamo_endpoint";
}

/// NATS Prometheus metric names
pub mod nats {
    /// Prefix for all NATS client metrics
    pub const PREFIX: &str = "nats_";

    /// ===== DistributedRuntime metrics =====
    /// Total number of bytes received by NATS client
    pub const IN_TOTAL_BYTES: &str = "nats_in_total_bytes";

    /// Total number of bytes sent by NATS client
    pub const OUT_OVERHEAD_BYTES: &str = "nats_out_overhead_bytes";

    /// Total number of messages received by NATS client
    pub const IN_MESSAGES: &str = "nats_in_messages";

    /// Total number of messages sent by NATS client
    pub const OUT_MESSAGES: &str = "nats_out_messages";

    /// Total number of connections established by NATS client
    pub const CONNECTS: &str = "nats_connects";

    /// Current connection state of NATS client (0=disconnected, 1=connected, 2=reconnecting)
    pub const CONNECTION_STATE: &str = "nats_connection_state";

    /// ===== Component metrics (ordered to match NatsStatsMetrics fields) =====
    /// Average processing time in milliseconds (maps to: average_processing_time in ms)
    pub const AVG_PROCESSING_MS: &str = "nats_avg_processing_time_ms";

    /// Total errors across all endpoints (maps to: num_errors)
    pub const TOTAL_ERRORS: &str = "nats_total_errors";

    /// Total requests across all endpoints (maps to: num_requests)
    pub const TOTAL_REQUESTS: &str = "nats_total_requests";

    /// Total processing time in milliseconds (maps to: processing_time in ms)
    pub const TOTAL_PROCESSING_MS: &str = "nats_total_processing_time_ms";

    /// Number of active services (derived from ServiceSet.services)
    pub const ACTIVE_SERVICES: &str = "nats_active_services";

    /// Number of active endpoints (derived from ServiceInfo.endpoints)
    pub const ACTIVE_ENDPOINTS: &str = "nats_active_endpoints";
}

/// All NATS client Prometheus metric names as an array for iteration/validation
pub const DRT_NATS_METRICS: &[&str] = &[
    nats::CONNECTION_STATE,
    nats::CONNECTS,
    nats::IN_TOTAL_BYTES,
    nats::IN_MESSAGES,
    nats::OUT_OVERHEAD_BYTES,
    nats::OUT_MESSAGES,
];

/// All component service Prometheus metric names as an array for iteration/validation
/// (ordered to match NatsStatsMetrics fields)
pub const COMPONENT_NATS_METRICS: &[&str] = &[
    nats::AVG_PROCESSING_MS,   // maps to: average_processing_time (nanoseconds)
    nats::TOTAL_ERRORS,        // maps to: num_errors
    nats::TOTAL_REQUESTS,      // maps to: num_requests
    nats::TOTAL_PROCESSING_MS, // maps to: processing_time (nanoseconds)
    nats::ACTIVE_SERVICES,     // derived from ServiceSet.services
    nats::ACTIVE_ENDPOINTS,    // derived from ServiceInfo.endpoints
];

/// Work handler Prometheus metric names
pub mod work_handler {
    /// Total number of requests processed by work handler
    pub const REQUESTS_TOTAL: &str = "requests_total";

    /// Total number of bytes received in requests by work handler
    pub const REQUEST_BYTES_TOTAL: &str = "request_bytes_total";

    /// Total number of bytes sent in responses by work handler
    pub const RESPONSE_BYTES_TOTAL: &str = "response_bytes_total";

    /// Number of requests currently being processed by work handler
    pub const CONCURRENT_REQUESTS: &str = "concurrent_requests";

    /// Time spent processing requests by work handler (histogram)
    pub const REQUEST_DURATION_SECONDS: &str = "request_duration_seconds";
}
