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

/// NATS client metrics. DistributedRuntime contains a NATS client shared by all children)
pub mod nats_client {
    /// Macro to generate NATS client metric names with the prefix
    macro_rules! nats_client_name {
        ($name:expr) => {
            concat!("nats_client_", $name)
        };
    }

    /// Prefix for all NATS client metrics
    pub const PREFIX: &str = nats_client_name!("");

    /// Total number of bytes received by NATS client
    pub const IN_TOTAL_BYTES: &str = nats_client_name!("in_total_bytes");

    /// Total number of bytes sent by NATS client
    pub const OUT_OVERHEAD_BYTES: &str = nats_client_name!("out_overhead_bytes");

    /// Total number of messages received by NATS client
    pub const IN_MESSAGES: &str = nats_client_name!("in_messages");

    /// Total number of messages sent by NATS client
    pub const OUT_MESSAGES: &str = nats_client_name!("out_messages");

    /// Total number of connections established by NATS client
    pub const CONNECTS: &str = nats_client_name!("connects");

    /// Current connection state of NATS client (0=disconnected, 1=connected, 2=reconnecting)
    pub const CONNECTION_STATE: &str = nats_client_name!("connection_state");
}

/// NATS service metrics, from the $SRV.STATS.<service_name> requests on NATS server
pub mod nats_service {
    /// Macro to generate NATS service metric names with the prefix
    macro_rules! nats_service_name {
        ($name:expr) => {
            concat!("nats_service_", $name)
        };
    }

    /// Prefix for all NATS service metrics
    pub const PREFIX: &str = nats_service_name!("");

    /// Average processing time in milliseconds (maps to: average_processing_time in ms)
    pub const AVG_PROCESSING_MS: &str = nats_service_name!("avg_processing_time_ms");

    /// Total errors across all endpoints (maps to: num_errors)
    pub const TOTAL_ERRORS: &str = nats_service_name!("total_errors");

    /// Total requests across all endpoints (maps to: num_requests)
    pub const TOTAL_REQUESTS: &str = nats_service_name!("total_requests");

    /// Total processing time in milliseconds (maps to: processing_time in ms)
    pub const TOTAL_PROCESSING_MS: &str = nats_service_name!("total_processing_time_ms");

    /// Number of active services (derived from ServiceSet.services)
    pub const ACTIVE_SERVICES: &str = nats_service_name!("active_services");

    /// Number of active endpoints (derived from ServiceInfo.endpoints)
    pub const ACTIVE_ENDPOINTS: &str = nats_service_name!("active_endpoints");
}

/// All NATS client Prometheus metric names as an array for iteration/validation
pub const DRT_NATS_METRICS: &[&str] = &[
    nats_client::CONNECTION_STATE,
    nats_client::CONNECTS,
    nats_client::IN_TOTAL_BYTES,
    nats_client::IN_MESSAGES,
    nats_client::OUT_OVERHEAD_BYTES,
    nats_client::OUT_MESSAGES,
];

/// All component service Prometheus metric names as an array for iteration/validation
/// (ordered to match NatsStatsMetrics fields)
pub const COMPONENT_NATS_METRICS: &[&str] = &[
    nats_service::AVG_PROCESSING_MS, // maps to: average_processing_time (nanoseconds)
    nats_service::TOTAL_ERRORS,      // maps to: num_errors
    nats_service::TOTAL_REQUESTS,    // maps to: num_requests
    nats_service::TOTAL_PROCESSING_MS, // maps to: processing_time (nanoseconds)
    nats_service::ACTIVE_SERVICES,   // derived from ServiceSet.services
    nats_service::ACTIVE_ENDPOINTS,  // derived from ServiceInfo.endpoints
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
    pub const INFLIGHT_REQUESTS: &str = "inflight_requests";

    /// Time spent processing requests by work handler (histogram)
    pub const REQUEST_DURATION_SECONDS: &str = "request_duration_seconds";
}

/// KVBM connector
pub mod kvbm_connector {
    /// KVBM connector leader
    pub const KVBM_CONNECTOR_LEADER: &str = "kvbm_connector_leader";

    /// KVBM connector worker
    pub const KVBM_CONNECTOR_WORKER: &str = "kvbm_connector_worker";
}
