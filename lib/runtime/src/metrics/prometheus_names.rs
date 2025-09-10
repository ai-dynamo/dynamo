// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metric name constants and sanitization utilities
//!
//! This module provides centralized Prometheus metric name constants and sanitization functions
//! for various components to ensure consistency and avoid duplication across the codebase.
//!
//! ## Key Differences: Prometheus Metric Names vs Prometheus Label Names
//!
//! **Metric names**: Allow colons and `__` anywhere. **Label names**: No colons, no `__` prefix.
//! Label names starting with `__` are reserved for Prometheus internal use.

use once_cell::sync::Lazy;
use regex::Regex;

/// Metric name prefixes used across the metrics system
pub mod name_prefix {
    /// Prefix for all Prometheus metric names.
    pub const COMPONENT: &str = "dynamo_component";

    /// Prefix for frontend service metrics
    pub const FRONTEND: &str = "dynamo_frontend";
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

/// Frontend service metrics (LLM HTTP service)
pub mod frontend_service {
    // TODO: Move DYN_METRICS_PREFIX and other environment variable names to environment_names.rs
    // for centralized environment variable constant management across the codebase
    /// Environment variable that overrides the default metric prefix
    pub const METRICS_PREFIX_ENV: &str = "DYN_METRICS_PREFIX";

    /// Total number of LLM requests processed
    pub const REQUESTS_TOTAL: &str = "requests_total";

    /// Number of inflight requests
    pub const INFLIGHT_REQUESTS: &str = "inflight_requests";

    /// Duration of LLM requests
    pub const REQUEST_DURATION_SECONDS: &str = "request_duration_seconds";

    /// Input sequence length in tokens
    pub const INPUT_SEQUENCE_TOKENS: &str = "input_sequence_tokens";

    /// Output sequence length in tokens
    pub const OUTPUT_SEQUENCE_TOKENS: &str = "output_sequence_tokens";

    /// Time to first token in seconds
    pub const TIME_TO_FIRST_TOKEN_SECONDS: &str = "time_to_first_token_seconds";

    /// Inter-token latency in seconds
    pub const INTER_TOKEN_LATENCY_SECONDS: &str = "inter_token_latency_seconds";

    /// Status label values
    pub mod status {
        /// Value for successful requests
        pub const SUCCESS: &str = "success";

        /// Value for failed requests
        pub const ERROR: &str = "error";
    }

    /// Request type label values
    pub mod request_type {
        /// Value for streaming requests
        pub const STREAM: &str = "stream";

        /// Value for unary requests
        pub const UNARY: &str = "unary";
    }
}

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

    /// Total number of errors in work handler processing
    pub const ERRORS_TOTAL: &str = "errors_total";

    /// Label name for error type classification
    pub const ERROR_TYPE_LABEL: &str = "error_type";

    /// Error type values for work handler metrics
    pub mod error_types {
        /// Deserialization error
        pub const DESERIALIZATION: &str = "deserialization";

        /// Invalid message format error
        pub const INVALID_MESSAGE: &str = "invalid_message";

        /// Response stream creation error
        pub const RESPONSE_STREAM: &str = "response_stream";

        /// Generation error
        pub const GENERATE: &str = "generate";

        /// Response publishing error
        pub const PUBLISH_RESPONSE: &str = "publish_response";

        /// Final message publishing error
        pub const PUBLISH_FINAL: &str = "publish_final";
    }
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

/// Task tracker Prometheus metric name suffixes
pub mod task_tracker {
    /// Total number of tasks issued/submitted
    pub const TASKS_ISSUED_TOTAL: &str = "tasks_issued_total";

    /// Total number of tasks started
    pub const TASKS_STARTED_TOTAL: &str = "tasks_started_total";

    /// Total number of successfully completed tasks
    pub const TASKS_SUCCESS_TOTAL: &str = "tasks_success_total";

    /// Total number of cancelled tasks
    pub const TASKS_CANCELLED_TOTAL: &str = "tasks_cancelled_total";

    /// Total number of failed tasks
    pub const TASKS_FAILED_TOTAL: &str = "tasks_failed_total";

    /// Total number of rejected tasks
    pub const TASKS_REJECTED_TOTAL: &str = "tasks_rejected_total";
}

/// DistributedRuntime core metrics
pub mod distributed_runtime {
    /// Total uptime of the DistributedRuntime in seconds
    pub const UPTIME_SECONDS: &str = "uptime_seconds";
}

/// KVBM connector
pub mod kvbm_connector {
    /// KVBM connector leader
    pub const KVBM_CONNECTOR_LEADER: &str = "kvbm_connector_leader";

    /// KVBM connector worker
    pub const KVBM_CONNECTOR_WORKER: &str = "kvbm_connector_worker";
}

/// KvStats metrics from LLM workers
pub mod kvstats {
    /// Macro to generate KvStats metric names with the prefix
    macro_rules! kvstats_name {
        ($name:expr) => {
            concat!("kvstats_", $name)
        };
    }

    /// Prefix for all KvStats metrics
    pub const PREFIX: &str = kvstats_name!("");

    /// Number of active KV cache blocks currently in use
    pub const ACTIVE_BLOCKS: &str = kvstats_name!("active_blocks");

    /// Total number of KV cache blocks available
    pub const TOTAL_BLOCKS: &str = kvstats_name!("total_blocks");

    /// GPU cache usage as a percentage (0.0-1.0)
    pub const GPU_CACHE_USAGE_PERCENT: &str = kvstats_name!("gpu_cache_usage_percent");

    /// GPU prefix cache hit rate as a percentage (0.0-1.0)
    pub const GPU_PREFIX_CACHE_HIT_RATE: &str = kvstats_name!("gpu_prefix_cache_hit_rate");
}

/// All KvStats Prometheus metric names as an array for iteration/validation
pub const KVSTATS_METRICS: &[&str] = &[
    kvstats::ACTIVE_BLOCKS,
    kvstats::TOTAL_BLOCKS,
    kvstats::GPU_CACHE_USAGE_PERCENT,
    kvstats::GPU_PREFIX_CACHE_HIT_RATE,
];

// KvRouter (including KvInexer) Prometheus metric names
pub mod kvrouter {
    /// Number of KV cache events applied to the index (including status)
    pub const KV_CACHE_EVENTS_APPLIED: &str = "kv_cache_events_applied";
}

// Shared regex patterns for Prometheus sanitization
static METRIC_INVALID_CHARS_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[^a-zA-Z0-9_:]").unwrap());
static LABEL_INVALID_CHARS_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[^a-zA-Z0-9_]").unwrap());
static INVALID_FIRST_CHAR_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[^a-zA-Z_]").unwrap());

/// Sanitizes a Prometheus metric name by converting invalid characters to underscores
/// and ensuring the first character is valid. Uses regex for clear validation.
/// Returns an error if the input cannot be sanitized into a valid name.
///
/// **Rules**: Pattern `[a-zA-Z_:][a-zA-Z0-9_:]*`. Allows colons and `__` anywhere.
pub fn sanitize_prometheus_name(raw: &str) -> anyhow::Result<String> {
    if raw.is_empty() {
        return Err(anyhow::anyhow!(
            "Cannot sanitize empty string into valid Prometheus name"
        ));
    }

    // Replace all invalid characters with underscores
    let mut sanitized = METRIC_INVALID_CHARS_PATTERN
        .replace_all(raw, "_")
        .to_string();

    // Ensure first character is valid (letter, underscore, or colon)
    if INVALID_FIRST_CHAR_PATTERN.is_match(&sanitized) {
        sanitized = format!("_{}", sanitized);
    }

    // Check if the result is all underscores (invalid input)
    if sanitized.chars().all(|c| c == '_') {
        return Err(anyhow::anyhow!(
            "Input '{}' contains only invalid characters and cannot be sanitized into a valid Prometheus name",
            raw
        ));
    }

    Ok(sanitized)
}

/// Sanitizes a Prometheus label name by converting invalid characters to underscores
/// and ensuring the first character is valid. Uses regex for clear validation.
/// Label names have stricter rules than metric names (no colons allowed).
/// Returns an error if the input cannot be sanitized into a valid label name.
///
/// **Rules**: Pattern `[a-zA-Z_][a-zA-Z0-9_]*`. No colons, no `__` prefix (reserved).
pub fn sanitize_prometheus_label(raw: &str) -> anyhow::Result<String> {
    if raw.is_empty() {
        return Err(anyhow::anyhow!(
            "Cannot sanitize empty string into valid Prometheus label"
        ));
    }

    // Replace all invalid characters with underscores (no colons allowed in labels)
    let mut sanitized = LABEL_INVALID_CHARS_PATTERN
        .replace_all(raw, "_")
        .to_string();

    // Ensure first character is valid (letter or underscore only)
    if INVALID_FIRST_CHAR_PATTERN.is_match(&sanitized) {
        sanitized = format!("_{}", sanitized);
    }

    // Prevent __ prefix (reserved for Prometheus internal use) but allow __ elsewhere
    if sanitized.starts_with("__") {
        sanitized = sanitized
            .strip_prefix("__")
            .unwrap_or(&sanitized)
            .to_string();
        if sanitized.is_empty() || !sanitized.chars().next().unwrap().is_ascii_alphabetic() {
            sanitized = format!("_{}", sanitized);
        }
    }

    // Check if the result is all underscores (invalid input)
    if sanitized.chars().all(|c| c == '_') {
        return Err(anyhow::anyhow!(
            "Input '{}' contains only invalid characters and cannot be sanitized into a valid Prometheus label",
            raw
        ));
    }

    Ok(sanitized)
}

/// Sanitizes a Prometheus frontend metric prefix by converting invalid characters to underscores
/// and ensuring the first character is valid. Uses the general prometheus name sanitization
/// but with frontend-specific fallback behavior.
pub fn sanitize_frontend_prometheus_prefix(raw: &str) -> String {
    if raw.is_empty() {
        return name_prefix::FRONTEND.to_string();
    }

    // Reuse the general prometheus name sanitization logic, fallback to frontend prefix on error
    sanitize_prometheus_name(raw).unwrap_or_else(|_| name_prefix::FRONTEND.to_string())
}

/// Builds a full component metric name by prepending the component prefix
/// Sanitizes the metric name to ensure it's valid for Prometheus
pub fn build_component_metric_name(metric_name: &str) -> String {
    let sanitized_name =
        sanitize_prometheus_name(metric_name).expect("metric name should be valid or sanitizable");
    format!("{}_{}", name_prefix::COMPONENT, sanitized_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_frontend_prometheus_prefix() {
        // Test that valid prefixes remain unchanged
        assert_eq!(
            sanitize_frontend_prometheus_prefix("dynamo_frontend"),
            "dynamo_frontend"
        );
        assert_eq!(
            sanitize_frontend_prometheus_prefix("custom_prefix"),
            "custom_prefix"
        );
        assert_eq!(sanitize_frontend_prometheus_prefix("test123"), "test123");

        // Test that invalid characters are converted to underscores
        assert_eq!(
            sanitize_frontend_prometheus_prefix("test prefix"),
            "test_prefix"
        );
        assert_eq!(
            sanitize_frontend_prometheus_prefix("test.prefix"),
            "test_prefix"
        );
        assert_eq!(
            sanitize_frontend_prometheus_prefix("test@prefix"),
            "test_prefix"
        );
        assert_eq!(
            sanitize_frontend_prometheus_prefix("test-prefix"),
            "test_prefix"
        );

        // Test that invalid first characters are fixed
        assert_eq!(sanitize_frontend_prometheus_prefix("123test"), "_123test");
        assert_eq!(sanitize_frontend_prometheus_prefix("@test"), "_test");

        // Test empty string fallback
        assert_eq!(
            sanitize_frontend_prometheus_prefix(""),
            name_prefix::FRONTEND
        );
    }

    #[test]
    fn test_sanitize_prometheus_name() {
        // Test that valid names remain unchanged
        assert_eq!(
            sanitize_prometheus_name("valid_name").unwrap(),
            "valid_name"
        );
        assert_eq!(sanitize_prometheus_name("test123").unwrap(), "test123");
        assert_eq!(
            sanitize_prometheus_name("test_name_123").unwrap(),
            "test_name_123"
        );
        assert_eq!(sanitize_prometheus_name("test:name").unwrap(), "test:name"); // colons allowed

        // Test that invalid characters are converted to underscores
        assert_eq!(sanitize_prometheus_name("test name").unwrap(), "test_name");
        assert_eq!(sanitize_prometheus_name("test.name").unwrap(), "test_name");
        assert_eq!(sanitize_prometheus_name("test@name").unwrap(), "test_name");
        assert_eq!(sanitize_prometheus_name("test-name").unwrap(), "test_name");
        assert_eq!(
            sanitize_prometheus_name("test$name#123").unwrap(),
            "test_name_123"
        );

        // Test that double underscores are ALLOWED in metric names (unlike labels)
        assert_eq!(
            sanitize_prometheus_name("test__name").unwrap(),
            "test__name"
        );
        assert_eq!(
            sanitize_prometheus_name("test___name").unwrap(),
            "test___name"
        );
        assert_eq!(sanitize_prometheus_name("__test").unwrap(), "__test"); // Leading double underscore OK

        // Test that invalid first characters are fixed
        assert_eq!(sanitize_prometheus_name("123test").unwrap(), "_123test");
        assert_eq!(sanitize_prometheus_name("@test").unwrap(), "_test"); // @ becomes _, no double underscore
        assert_eq!(sanitize_prometheus_name("-test").unwrap(), "_test"); // - becomes _, no double underscore
        assert_eq!(sanitize_prometheus_name(".test").unwrap(), "_test"); // . becomes _, no double underscore

        // Test empty string returns error
        assert!(sanitize_prometheus_name("").is_err());

        // Test complex cases
        assert_eq!(
            sanitize_prometheus_name("123.test-name@domain").unwrap(),
            "_123_test_name_domain"
        );

        // Test that strings with only invalid characters return error
        assert!(sanitize_prometheus_name("@#$%").is_err());
        assert!(sanitize_prometheus_name("!!!!").is_err());
    }

    #[test]
    fn test_sanitize_prometheus_label() {
        // Test that valid labels remain unchanged
        assert_eq!(
            sanitize_prometheus_label("valid_label").unwrap(),
            "valid_label"
        );
        assert_eq!(sanitize_prometheus_label("test123").unwrap(), "test123");
        assert_eq!(
            sanitize_prometheus_label("test_label_123").unwrap(),
            "test_label_123"
        );

        // Test that colons are NOT allowed in labels (stricter than names)
        assert_eq!(
            sanitize_prometheus_label("test:label").unwrap(),
            "test_label"
        );

        // Test that invalid characters are converted to underscores
        assert_eq!(
            sanitize_prometheus_label("test label").unwrap(),
            "test_label"
        );
        assert_eq!(
            sanitize_prometheus_label("test.label").unwrap(),
            "test_label"
        );
        assert_eq!(
            sanitize_prometheus_label("test@label").unwrap(),
            "test_label"
        );
        assert_eq!(
            sanitize_prometheus_label("test-label").unwrap(),
            "test_label"
        );
        assert_eq!(
            sanitize_prometheus_label("test$label#123").unwrap(),
            "test_label_123"
        );

        // Test that double underscores are ALLOWED in middle but NOT at start
        assert_eq!(
            sanitize_prometheus_label("test__label").unwrap(),
            "test__label"
        ); // OK in middle
        assert_eq!(
            sanitize_prometheus_label("test___label").unwrap(),
            "test___label"
        ); // OK in middle
        assert_eq!(
            sanitize_prometheus_label("test____label").unwrap(),
            "test____label"
        ); // OK in middle
        assert_eq!(sanitize_prometheus_label("__test").unwrap(), "test"); // Leading __ removed
        assert!(sanitize_prometheus_label("____").is_err()); // All underscores should error

        // Test that invalid first characters are fixed (no colons allowed)
        assert_eq!(sanitize_prometheus_label("123test").unwrap(), "_123test");
        assert_eq!(sanitize_prometheus_label("@test").unwrap(), "_test");
        assert_eq!(sanitize_prometheus_label(":test").unwrap(), "_test"); // colon not allowed
        assert_eq!(sanitize_prometheus_label("-test").unwrap(), "_test");

        // Test empty string returns error
        assert!(sanitize_prometheus_label("").is_err());

        // Test complex cases
        assert_eq!(
            sanitize_prometheus_label("123:test-label@domain").unwrap(),
            "_123_test_label_domain"
        );

        // Test that strings with only invalid characters return error
        assert!(sanitize_prometheus_label("@#$%").is_err()); // @#$% -> ____ -> ___ -> all underscores error
        assert!(sanitize_prometheus_label("!!!!").is_err()); // !!!! -> ____ -> ___ -> all underscores error
    }

    #[test]
    fn test_build_component_metric_name() {
        // Test that valid names work correctly
        assert_eq!(
            build_component_metric_name("test_metric"),
            "dynamo_component_test_metric"
        );
        assert_eq!(
            build_component_metric_name("requests_total"),
            "dynamo_component_requests_total"
        );

        // Test that invalid characters are sanitized
        assert_eq!(
            build_component_metric_name("test metric"),
            "dynamo_component_test_metric"
        );
        assert_eq!(
            build_component_metric_name("test.metric"),
            "dynamo_component_test_metric"
        );
        assert_eq!(
            build_component_metric_name("test@metric"),
            "dynamo_component_test_metric"
        );

        // Test that invalid first characters are fixed
        assert_eq!(
            build_component_metric_name("123metric"),
            "dynamo_component__123metric"
        );
    }

    #[test]
    #[should_panic(expected = "metric name should be valid or sanitizable")]
    fn test_build_component_metric_name_panics_on_invalid_input() {
        // Test that completely invalid input panics with clear message
        build_component_metric_name("@#$%");
    }

    #[test]
    #[should_panic(expected = "metric name should be valid or sanitizable")]
    fn test_build_component_metric_name_panics_on_empty_input() {
        // Test that empty input panics with clear message
        build_component_metric_name("");
    }
}
