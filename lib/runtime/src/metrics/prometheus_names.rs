// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metric name constants and sanitization utilities
//!
//! This module provides centralized Prometheus metric name constants and sanitization functions
//! for various components to ensure consistency and avoid duplication across the codebase.
//!
//! ⚠️  **CRITICAL: REGENERATE PYTHON FILE AFTER CHANGES** ⚠️
//! When modifying constants in this file, regenerate the Python module:
//!     cargo run -p dynamo-codegen --bin gen-python-prometheus-names
//!
//! This generates `lib/bindings/python/src/dynamo/prometheus_names.py`
//! with pure Python constants (no Rust bindings needed).
//!
//! ## Naming Conventions
//!
//! All metric names should follow: `{prefix}_{name}_{suffix}`
//!
//! **Prefix**: Component identifier (`dynamo_component_`, `dynamo_frontend_`, etc.)
//! **Name**: Descriptive snake_case name indicating what is measured
//! **Suffix**:
//!   - Units: `_seconds`, `_bytes`, `_ms`, `_percent`, `_messages`, `_connections`
//!   - Counters: `_total` (not `total_` prefix) - for cumulative metrics that only increase
//!   - Gauges: No `_total` suffix - for current state metrics that can go up and down
//!   - Note: Do not use `_counter`, `_gauge`, `_time`, or `_size` in Prometheus names (too vague)
//!
//! **Common Transformations**:
//! - ❌ `_counter` → ✅ `_total`
//! - ❌ `_sum` → ✅ `_total`
//! - ❌ `_gauge` → ✅ (no suffix needed for current values)
//! - ❌ `_time` → ✅ `_seconds`, `_ms`, `_hours`, `_duration_seconds`
//! - ❌ `_time_total` → ✅ `_seconds_total`, `_ms_total`, `_hours_total`
//! - ❌ `_total_time` → ✅ `_seconds_total`, `_ms_total`, `_hours_total`
//! - ❌ `_total_time_seconds` → ✅ `_seconds_total`
//! - ❌ `_average_time` → ✅ `_seconds_avg`, `_ms_avg`
//! - ❌ `_size` → ✅ `_bytes`, `_total`, `_length`
//! - ❌ `_some_request_size` → ✅ `_some_request_bytes_avg`
//! - ❌ `_rate` → ✅ `_per_second`, `_per_minute`
//! - ❌ `disconnected_clients_total` → ✅ `disconnected_clients` (gauge, not counter)
//! - ❌ `inflight_requests_total` → ✅ `inflight_requests` (gauge, not counter)
//! - ❌ `connections_total` → ✅ `current_connections` (gauge, not counter)
//!
//! **Examples**:
//! - ✅ `dynamo_frontend_requests_total` - Total request counter (not `incoming_requests`)
//! - ✅ `dynamo_frontend_request_duration_seconds` - Request duration histogram (not `response_time`)
//! - ✅ `dynamo_component_errors_total` - Total error counter (not `total_errors`)
//! - ✅ `dynamo_component_memory_usage_bytes` - Memory usage gauge
//! - ✅ `dynamo_frontend_inflight_requests` - Current inflight requests gauge
//! - ✅ `nats_client_connection_duration_ms` - Connection time in milliseconds
//! - ✅ `dynamo_component_cpu_usage_percent` - CPU usage percentage
//! - ✅ `dynamo_frontend_tokens_per_second` - Token generation rate
//! - ✅ `nats_client_current_connections` - Current active connections gauge
//! - ✅ `nats_client_in_messages` - Total messages received counter
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
///
/// ⚠️  Python codegen: Run gen-python-prometheus-names after changes
pub mod frontend_service {
    // TODO: Move DYN_METRICS_PREFIX and other environment variable names to environment_names.rs
    // for centralized environment variable constant management across the codebase
    /// Environment variable that overrides the default metric prefix
    pub const METRICS_PREFIX_ENV: &str = "DYN_METRICS_PREFIX";

    /// Total number of LLM requests processed
    pub const REQUESTS_TOTAL: &str = "requests_total";

    /// Number of requests waiting in HTTP queue before receiving the first response (gauge)
    pub const QUEUED_REQUESTS: &str = "queued_requests";

    /// Number of inflight/concurrent requests going to the engine (vLLM, SGLang, ...)
    /// Note: This is a gauge metric (current state) that can go up and down, so no _total suffix
    pub const INFLIGHT_REQUESTS: &str = "inflight_requests";

    /// Number of disconnected clients (gauge that can go up and down)
    pub const DISCONNECTED_CLIENTS: &str = "disconnected_clients";

    /// Duration of LLM requests
    pub const REQUEST_DURATION_SECONDS: &str = "request_duration_seconds";

    /// Input sequence length in tokens
    pub const INPUT_SEQUENCE_TOKENS: &str = "input_sequence_tokens";

    /// Output sequence length in tokens
    pub const OUTPUT_SEQUENCE_TOKENS: &str = "output_sequence_tokens";

    /// Number of cached tokens (prefix cache hits) per request
    pub const CACHED_TOKENS: &str = "cached_tokens";

    /// Total number of output tokens generated (counter that updates in real-time)
    pub const OUTPUT_TOKENS_TOTAL: &str = "output_tokens_total";

    /// Time to first token in seconds
    pub const TIME_TO_FIRST_TOKEN_SECONDS: &str = "time_to_first_token_seconds";

    /// Inter-token latency in seconds
    pub const INTER_TOKEN_LATENCY_SECONDS: &str = "inter_token_latency_seconds";

    /// Model configuration metrics
    ///
    /// Runtime config metrics (from ModelRuntimeConfig):
    /// Total KV blocks available for a worker serving the model
    pub const MODEL_TOTAL_KV_BLOCKS: &str = "model_total_kv_blocks";

    /// Maximum number of sequences for a worker serving the model (runtime config)
    pub const MODEL_MAX_NUM_SEQS: &str = "model_max_num_seqs";

    /// Maximum number of batched tokens for a worker serving the model (runtime config)
    pub const MODEL_MAX_NUM_BATCHED_TOKENS: &str = "model_max_num_batched_tokens";

    /// MDC metrics (from ModelDeploymentCard):
    /// Maximum context length for a worker serving the model (MDC)
    pub const MODEL_CONTEXT_LENGTH: &str = "model_context_length";

    /// KV cache block size for a worker serving the model (MDC)
    pub const MODEL_KV_CACHE_BLOCK_SIZE: &str = "model_kv_cache_block_size";

    /// Request migration limit for a worker serving the model (MDC)
    pub const MODEL_MIGRATION_LIMIT: &str = "model_migration_limit";

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
    /// Note: This is a gauge metric (current state) that can go up and down, so no _total suffix
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

/// KVBM
pub mod kvbm {
    /// The number of offload blocks from device to host
    pub const OFFLOAD_BLOCKS_D2H: &str = "offload_blocks_d2h";

    /// The number of offload blocks from host to disk
    pub const OFFLOAD_BLOCKS_H2D: &str = "offload_blocks_h2d";

    /// The number of offload blocks from device to disk (bypassing host memory)
    pub const OFFLOAD_BLOCKS_D2D: &str = "offload_blocks_d2d";

    /// The number of onboard blocks from host to device
    pub const ONBOARD_BLOCKS_H2D: &str = "onboard_blocks_h2d";

    /// The number of onboard blocks from disk to device
    pub const ONBOARD_BLOCKS_D2D: &str = "onboard_blocks_d2d";

    /// The number of matched tokens
    pub const MATCHED_TOKENS: &str = "matched_tokens";

    /// Host cache hit rate (0.0-1.0) from the sliding window
    pub const HOST_CACHE_HIT_RATE: &str = "host_cache_hit_rate";

    /// Disk cache hit rate (0.0-1.0) from the sliding window
    pub const DISK_CACHE_HIT_RATE: &str = "disk_cache_hit_rate";
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

/// Safely converts a u64 value to i64 for Prometheus metrics
///
/// Since Prometheus IntGaugeVec uses i64 but our data types use u64,
/// this function clamps large u64 values to i64::MAX to prevent overflow
/// and ensure metrics remain positive.
///
/// # Arguments
/// * `value` - The u64 value to convert
///
/// # Returns
/// An i64 value, clamped to i64::MAX if the input exceeds i64::MAX
///
/// # Examples
/// ```
/// use dynamo_runtime::metrics::prometheus_names::clamp_u64_to_i64;
///
/// assert_eq!(clamp_u64_to_i64(100), 100);
/// assert_eq!(clamp_u64_to_i64(u64::MAX), i64::MAX);
/// ```
pub fn clamp_u64_to_i64(value: u64) -> i64 {
    if value > i64::MAX as u64 {
        i64::MAX
    } else {
        value as i64
    }
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

    #[test]
    fn test_clamp_u64_to_i64() {
        // Test normal values within i64 range
        assert_eq!(clamp_u64_to_i64(0), 0);
        assert_eq!(clamp_u64_to_i64(100), 100);
        assert_eq!(clamp_u64_to_i64(1000000), 1000000);

        // Test maximum i64 value
        assert_eq!(clamp_u64_to_i64(i64::MAX as u64), i64::MAX);

        // Test values that exceed i64::MAX
        assert_eq!(clamp_u64_to_i64(u64::MAX), i64::MAX);
        assert_eq!(clamp_u64_to_i64((i64::MAX as u64) + 1), i64::MAX);
        assert_eq!(clamp_u64_to_i64((i64::MAX as u64) + 1000), i64::MAX);
    }
}
