// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Prometheus metric name constants
//!
//! ⚠️  **CRITICAL: SYNC WITH RUST SOURCE AND PYTHON TYPE STUBS** ⚠️
//! This file exposes constants from `lib/runtime/src/metrics/prometheus_names.rs` to Python.
//! When the source file is modified, you MUST update BOTH files to match:
//!
//! 1. **This Rust file** - Update the actual Python bindings implementation
//! 2. **Python type stubs** - Update `lib/bindings/python/src/dynamo/_core.pyi`
//!    The .pyi file provides type hints for IDEs and static type checkers.
//!    Without updating it, IDEs won't recognize new classes/methods for autocomplete.
//!
//! The constants here should mirror the structure and values from the Rust source.
//! Any changes to metric names in the source must be reflected here immediately.
//!
//! Files to sync:
//! - Source:      `lib/runtime/src/metrics/prometheus_names.rs`
//! - This file:   `lib/bindings/python/rust/prometheus_names.rs`
//! - Type stubs:  `lib/bindings/python/src/dynamo/_core.pyi`
//!
//! ## Python Usage Example
//!
//! ```python
//! from dynamo._core import prometheus_names
//!
//! # Access metrics directly (no constructor call needed!)
//! frontend = prometheus_names.frontend
//! print(frontend.requests_total)           # "dynamo_frontend_requests_total"
//! print(frontend.queued_requests)          # "dynamo_frontend_queued_requests"
//! print(frontend.inflight_requests)        # "dynamo_frontend_inflight_requests"
//! print(frontend.disconnected_clients)     # "dynamo_frontend_disconnected_clients"
//! print(frontend.request_duration_seconds) # "dynamo_frontend_request_duration_seconds"
//! print(frontend.input_sequence_tokens)    # "dynamo_frontend_input_sequence_tokens"
//! print(frontend.output_sequence_tokens)   # "dynamo_frontend_output_sequence_tokens"
//! print(frontend.time_to_first_token_seconds) # "dynamo_frontend_time_to_first_token_seconds"
//! print(frontend.inter_token_latency_seconds) # "dynamo_frontend_inter_token_latency_seconds"
//! print(frontend.model_context_length)     # "dynamo_frontend_model_context_length"
//! print(frontend.model_kv_cache_block_size) # "dynamo_frontend_model_kv_cache_block_size"
//! print(frontend.model_migration_limit)    # "dynamo_frontend_model_migration_limit"
//!
//! work_handler = prometheus_names.work_handler
//! print(work_handler.requests_total)       # "dynamo_component_requests_total"
//! print(work_handler.request_bytes_total)  # "dynamo_component_request_bytes_total"
//! print(work_handler.response_bytes_total) # "dynamo_component_response_bytes_total"
//! print(work_handler.inflight_requests)    # "dynamo_component_inflight_requests"
//! print(work_handler.request_duration_seconds) # "dynamo_component_request_duration_seconds"
//! print(work_handler.errors_total)         # "dynamo_component_errors_total"
//!
//! kvstats = prometheus_names.kvstats
//! print(kvstats.active_blocks)             # "kvstats_active_blocks"
//! print(kvstats.total_blocks)              # "kvstats_total_blocks"
//! print(kvstats.gpu_cache_usage_percent)   # "kvstats_gpu_cache_usage_percent"
//! print(kvstats.gpu_prefix_cache_hit_rate) # "kvstats_gpu_prefix_cache_hit_rate"
//!
//! # Use in Prometheus queries
//! query = f"rate({frontend.requests_total}[5m])"
//! pattern = rf'{work_handler.requests_total}\{{[^}}]*model="[^"]*"[^}}]*\}}'
//! ```

use dynamo_runtime::metrics::prometheus_names::*;
use pyo3::prelude::*;

/// Main container for all Prometheus metric name constants
#[pyclass]
pub struct PrometheusNames;

#[pymethods]
impl PrometheusNames {
    /// Frontend service metrics
    #[getter]
    fn frontend(&self) -> FrontendService {
        FrontendService
    }

    /// Work handler metrics
    #[getter]
    fn work_handler(&self) -> WorkHandler {
        WorkHandler
    }

    /// KV stats metrics
    #[getter]
    fn kvstats(&self) -> KvStatsMetrics {
        KvStatsMetrics
    }
}

/// Frontend service metrics (LLM HTTP service)
/// These methods return the full metric names with the "dynamo_frontend_" prefix
///
/// Note: We use instance methods instead of static methods for better Python ergonomics
/// - The `concat!` macro only accepts string literals, not const references
/// - We need to combine `name_prefix::FRONTEND` + `frontend_service::*` constants at runtime
/// - This ensures we use actual Rust constants rather than hardcoded literals
#[pyclass]
pub struct FrontendService;

#[pymethods]
impl FrontendService {
    /// Total number of LLM requests processed
    #[getter]
    fn requests_total(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::REQUESTS_TOTAL
        )
    }

    /// Number of requests waiting in HTTP queue before receiving the first response
    #[getter]
    fn queued_requests(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::QUEUED_REQUESTS
        )
    }

    /// Number of inflight requests going to the engine (vLLM, SGLang, ...)
    #[getter]
    fn inflight_requests(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::INFLIGHT_REQUESTS
        )
    }

    /// Duration of LLM requests
    #[getter]
    fn request_duration_seconds(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::REQUEST_DURATION_SECONDS
        )
    }

    /// Input sequence length in tokens
    #[getter]
    fn input_sequence_tokens(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::INPUT_SEQUENCE_TOKENS
        )
    }

    /// Output sequence length in tokens
    #[getter]
    fn output_sequence_tokens(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::OUTPUT_SEQUENCE_TOKENS
        )
    }

    /// Time to first token in seconds
    #[getter]
    fn time_to_first_token_seconds(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::TIME_TO_FIRST_TOKEN_SECONDS
        )
    }

    /// Inter-token latency in seconds
    #[getter]
    fn inter_token_latency_seconds(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::INTER_TOKEN_LATENCY_SECONDS
        )
    }

    /// Number of disconnected clients
    #[getter]
    fn disconnected_clients(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::DISCONNECTED_CLIENTS
        )
    }

    /// Model total KV blocks
    #[getter]
    fn model_total_kv_blocks(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::MODEL_TOTAL_KV_BLOCKS
        )
    }

    /// Model max number of sequences
    #[getter]
    fn model_max_num_seqs(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::MODEL_MAX_NUM_SEQS
        )
    }

    /// Model max number of batched tokens
    #[getter]
    fn model_max_num_batched_tokens(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::MODEL_MAX_NUM_BATCHED_TOKENS
        )
    }

    /// Model context length
    #[getter]
    fn model_context_length(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::MODEL_CONTEXT_LENGTH
        )
    }

    /// Model KV cache block size
    #[getter]
    fn model_kv_cache_block_size(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::MODEL_KV_CACHE_BLOCK_SIZE
        )
    }

    /// Model migration limit
    #[getter]
    fn model_migration_limit(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::FRONTEND,
            frontend_service::MODEL_MIGRATION_LIMIT
        )
    }
}

/// Work handler metrics (component request processing)
/// These methods return the full metric names with the "dynamo_component_" prefix
#[pyclass]
pub struct WorkHandler;

#[pymethods]
impl WorkHandler {
    /// Total number of requests processed by work handler
    #[getter]
    fn requests_total(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::COMPONENT,
            work_handler::REQUESTS_TOTAL
        )
    }

    /// Total number of bytes received in requests by work handler
    #[getter]
    fn request_bytes_total(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::COMPONENT,
            work_handler::REQUEST_BYTES_TOTAL
        )
    }

    /// Total number of bytes sent in responses by work handler
    #[getter]
    fn response_bytes_total(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::COMPONENT,
            work_handler::RESPONSE_BYTES_TOTAL
        )
    }

    /// Number of requests currently being processed by work handler
    #[getter]
    fn inflight_requests(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::COMPONENT,
            work_handler::INFLIGHT_REQUESTS
        )
    }

    /// Time spent processing requests by work handler (histogram)
    #[getter]
    fn request_duration_seconds(&self) -> String {
        format!(
            "{}_{}",
            name_prefix::COMPONENT,
            work_handler::REQUEST_DURATION_SECONDS
        )
    }

    /// Total number of errors in work handler processing
    #[getter]
    fn errors_total(&self) -> String {
        format!("{}_{}", name_prefix::COMPONENT, work_handler::ERRORS_TOTAL)
    }
}

/// KV stats metrics (KV cache statistics)
/// These methods return the metric names with the "kvstats_" prefix
#[pyclass]
pub struct KvStatsMetrics;

#[pymethods]
impl KvStatsMetrics {
    /// Number of active KV cache blocks currently in use
    #[getter]
    fn active_blocks(&self) -> String {
        kvstats::ACTIVE_BLOCKS.to_string()
    }

    /// Total number of KV cache blocks available
    #[getter]
    fn total_blocks(&self) -> String {
        kvstats::TOTAL_BLOCKS.to_string()
    }

    /// GPU cache usage as a percentage (0.0-1.0)
    #[getter]
    fn gpu_cache_usage_percent(&self) -> String {
        kvstats::GPU_CACHE_USAGE_PERCENT.to_string()
    }

    /// GPU prefix cache hit rate as a percentage (0.0-1.0)
    #[getter]
    fn gpu_prefix_cache_hit_rate(&self) -> String {
        kvstats::GPU_PREFIX_CACHE_HIT_RATE.to_string()
    }
}

/// Add prometheus_names module to the Python bindings
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PrometheusNames>()?;
    m.add_class::<FrontendService>()?;
    m.add_class::<WorkHandler>()?;
    m.add_class::<KvStatsMetrics>()?;

    // Add a module-level singleton instance for convenience
    let prometheus_names_instance = PrometheusNames;
    m.add("prometheus_names", prometheus_names_instance)?;

    Ok(())
}
