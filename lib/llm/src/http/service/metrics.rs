// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, sse::Event},
    routing::get,
};
use dynamo_runtime::{
    config::environment_names::llm::metrics as env_metrics,
    metrics::prometheus_names::{
        frontend_service, name_prefix, sanitize_frontend_prometheus_prefix,
    },
};
use prometheus::{Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts};
use serde::Serialize;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::metrics::prometheus_names::clamp_u64_to_i64;

pub use prometheus::Registry;

use super::RouteDoc;

/// Generate log-spaced histogram buckets with values rounded to 2 significant figures.
///
/// # Arguments
/// * `min` - Minimum value for the buckets (must be > 0 for log spacing)
/// * `max` - Maximum value for the buckets (must be > min)
/// * `count` - Number of buckets to generate
///
/// # Returns
/// A vector of log-spaced values, always starting with 0.0 and ending with the rounded max value.
/// Duplicates created by rounding are removed, so the final count may be less than requested.
///
/// # Note
/// With 2 significant figures, there are roughly 90 unique values per order of magnitude.
/// Requesting more buckets than can be uniquely represented will result in deduplication.
fn generate_log_buckets(min: f64, max: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return vec![];
    }
    if count == 1 {
        return vec![0.0];
    }

    let requested_count = count;
    let mut buckets = Vec::with_capacity(count);
    buckets.push(0.0);

    // Generate log-spaced values from min to max
    for i in 1..count {
        let log_min = min.ln();
        let log_max = max.ln();
        let log_value = log_min + (log_max - log_min) * (i as f64) / ((count - 1) as f64);
        let value = log_value.exp();
        buckets.push(round_to_sig_figs(value, 2));
    }

    // Remove consecutive duplicates (buckets are already sorted)
    let original_len = buckets.len();
    buckets.dedup();

    // Warn if significant deduplication occurred
    if buckets.len() < original_len && (original_len - buckets.len()) > original_len / 10 {
        tracing::warn!(
            requested = requested_count,
            unique = buckets.len(),
            duplicates = original_len - buckets.len(),
            min = min,
            max = max,
            "Histogram bucket generation: Significant duplicate values after rounding to 2 sig figs. \
             Consider reducing bucket count or increasing range."
        );
    }

    buckets
}

/// Round a number to a specified number of significant figures
fn round_to_sig_figs(value: f64, sig_figs: u32) -> f64 {
    if value == 0.0 {
        return 0.0;
    }

    let magnitude = value.abs().log10().floor();
    let scale = 10_f64.powf(sig_figs as f64 - 1.0 - magnitude);
    (value * scale).round() / scale
}

const MAX_BUCKET_COUNT: usize = 512;

fn validate_bucket_config(min: f64, max: f64, count: usize) -> bool {
    min.is_finite()
        && max.is_finite()
        && min > 0.0
        && min < max
        && count > 0
        && count <= MAX_BUCKET_COUNT
}

/// Parse histogram bucket configuration from environment variables
/// Returns (min, max, count) with defaults if not specified
fn parse_bucket_config(
    env_prefix: &str,
    default_min: f64,
    default_max: f64,
    default_count: usize,
) -> (f64, f64, usize) {
    if !validate_bucket_config(default_min, default_max, default_count) {
        tracing::error!(
            default_min,
            default_max,
            default_count,
            "Invalid default histogram configuration"
        );
        return (1.0, 10.0, 10);
    }
    let env_prefix = format!("{}{}", env_metrics::HISTOGRAM_PREFIX, env_prefix);
    let mut min = std::env::var(format!("{env_prefix}_MIN"))
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default_min);
    let mut max = std::env::var(format!("{env_prefix}_MAX"))
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default_max);
    let mut count = std::env::var(format!("{env_prefix}_COUNT"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default_count);

    if !validate_bucket_config(min, max, count) {
        tracing::warn!(
            min=%min,
            max=%max,
            count=%count,
            "Invalid histogram configuration given, using defaults"
        );
        min = default_min;
        max = default_max;
        count = default_count;
    }

    (min, max, count)
}

/// State for metrics handler with custom backend support
struct MetricsHandlerState {
    registry: Arc<Registry>,
}

pub struct Metrics {
    request_counter: IntCounterVec,
    inflight_gauge: IntGaugeVec,
    client_disconnect_gauge: prometheus::IntGauge,
    http_queue_gauge: IntGaugeVec,
    request_duration: HistogramVec,
    input_sequence_length: HistogramVec,
    output_sequence_length: HistogramVec,
    cached_tokens: HistogramVec,
    output_tokens_counter: IntCounterVec,
    time_to_first_token: HistogramVec,
    inter_token_latency: HistogramVec,

    // Runtime configuration metrics. Note: Some of these metrics represent counter-like values from
    // source systems, but are implemented as gauges because they are copied/synchronized from upstream
    // counter values rather than being directly incremented.
    model_total_kv_blocks: IntGaugeVec,
    model_max_num_seqs: IntGaugeVec,
    model_max_num_batched_tokens: IntGaugeVec,
    model_context_length: IntGaugeVec,
    model_kv_cache_block_size: IntGaugeVec,
    model_migration_limit: IntGaugeVec,
    model_migration_total: IntCounterVec,
}

// Inflight tracks requests from HTTP handler start until complete response is finished.
// HTTP queue tracks requests from HTTP handler start until first token generation begins (including prefill time).
// HTTP queue time is a subset of inflight time. For detailed explanation, see:
// deploy/metrics/README.md - "Request Processing Flow" section

/// RAII object for HTTP queue gauge
/// Tracks requests from HTTP handler start until metrics processing begins
pub struct HttpQueueGuard {
    metrics: Arc<Metrics>,
    model: String,
}

/// RAII object for inflight gauge and request counters
/// If this object is dropped without calling `mark_ok`, then the request will increment
/// the request counter with the `status` label with [`frontend_service::status::ERROR`]; otherwise, it will increment
/// the counter with `status` label [`frontend_service::status::SUCCESS`]
pub struct InflightGuard {
    metrics: Arc<Metrics>,
    model: String,
    endpoint: Endpoint,
    request_type: RequestType,
    status: Status,
    timer: Instant,
}

/// Requests will be logged by the type of endpoint hit
/// This will include llamastack in the future
pub enum Endpoint {
    /// OAI Completions
    Completions,

    /// OAI Chat Completions
    ChatCompletions,

    /// OAI Embeddings
    Embeddings,

    /// OAI Responses
    Responses,

    /// Tensor
    Tensor,
}

/// Metrics for the HTTP service
pub enum RequestType {
    /// SingleIn / SingleOut
    Unary,

    /// SingleIn / ManyOut
    Stream,
}

/// Status
#[derive(PartialEq)]
pub enum Status {
    Success,
    Error,
}

/// Track response-specific metrics
pub struct ResponseMetricCollector {
    metrics: Arc<Metrics>,
    model: String,
    start_time: Instant,
    // we use is_first_token to distinguish TTFT from ITL. It is true by default and
    // flipped to false when the first token is returned and TTFT is published.
    is_first_token: bool,
    // we track the last response time so that ITL for the newly returned tokens can
    // be computed.
    last_response_time: Option<Duration>,
    osl: usize,
    // we track if cached_tokens has been observed to ensure we only increment once per request
    cached_tokens_observed: bool,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Create Metrics with the standard prefix defined by [`name_prefix::FRONTEND`] or specify custom prefix via the following environment variable:
    /// - `DYN_METRICS_PREFIX`: Override the default metrics prefix
    ///
    /// The following metrics will be created with the configured prefix:
    /// - `{prefix}_requests_total` - IntCounterVec for the total number of requests processed
    /// - `{prefix}_inflight_requests` - IntGaugeVec for the number of inflight/concurrent requests
    /// - `{prefix}_disconnected_clients` - IntGauge for the number of disconnected clients
    /// - `{prefix}_request_duration_seconds` - HistogramVec for the duration of requests
    /// - `{prefix}_input_sequence_tokens` - HistogramVec for input sequence length in tokens
    /// - `{prefix}_output_sequence_tokens` - HistogramVec for output sequence length in tokens
    /// - `{prefix}_output_tokens_total` - IntCounterVec for total output tokens generated (real-time updates)
    /// - `{prefix}_time_to_first_token_seconds` - HistogramVec for time to first token in seconds
    /// - `{prefix}_inter_token_latency_seconds` - HistogramVec for inter-token latency in seconds
    ///
    /// ## Histogram Bucket Configuration
    ///
    /// All histograms use log-spaced buckets rounded to 2 significant figures. Bucket configuration
    /// can be customized via environment variables (MIN: minimum value, MAX: maximum value, COUNT: number of buckets):
    ///
    /// - `DYN_METRICS_REQUEST_DURATION_{MIN,MAX,COUNT}` - Request duration histogram (defaults: 1.0, 256.0, 10)
    /// - `DYN_METRICS_INPUT_SEQUENCE_{MIN,MAX,COUNT}` - Input sequence length histogram (defaults: 50.0, 128000.0, 12)
    /// - `DYN_METRICS_OUTPUT_SEQUENCE_{MIN,MAX,COUNT}` - Output sequence length histogram (defaults: 50.0, 32000.0, 10)
    /// - `DYN_METRICS_TTFT_{MIN,MAX,COUNT}` - Time to first token histogram (defaults: 0.001, 480.0, 18)
    /// - `DYN_METRICS_ITL_{MIN,MAX,COUNT}` - Inter-token latency histogram (defaults: 0.001, 2.0, 13)
    ///
    /// ## Model Configuration Metrics
    ///
    /// Runtime config metrics (from ModelRuntimeConfig):
    /// - `{prefix}_model_total_kv_blocks` - IntGaugeVec for total KV cache blocks available for a worker serving the model
    /// - `{prefix}_model_max_num_seqs` - IntGaugeVec for maximum sequences for a worker serving the model
    /// - `{prefix}_model_max_num_batched_tokens` - IntGaugeVec for maximum batched tokens for a worker serving the model
    ///
    /// MDC metrics (from ModelDeploymentCard):
    /// - `{prefix}_model_context_length` - IntGaugeVec for maximum context length for a worker serving the model
    /// - `{prefix}_model_kv_cache_block_size` - IntGaugeVec for KV cache block size for a worker serving the model
    /// - `{prefix}_model_migration_limit` - IntGaugeVec for request migration limit for a worker serving the model
    ///
    /// ## Runtime Config Polling Configuration
    ///
    /// The polling behavior can be configured via environment variables:
    /// - `DYN_HTTP_SVC_CONFIG_METRICS_POLL_INTERVAL_SECS`: Poll interval in seconds (must be > 0, supports fractional seconds, defaults to 8)
    ///
    /// Metrics are never removed to preserve historical data. Runtime config and MDC
    /// metrics are updated when models are discovered and their configurations are available.
    pub fn new() -> Self {
        let raw_prefix = std::env::var(env_metrics::DYN_METRICS_PREFIX)
            .unwrap_or_else(|_| name_prefix::FRONTEND.to_string());
        let prefix = sanitize_frontend_prometheus_prefix(&raw_prefix);
        if prefix != raw_prefix {
            tracing::warn!(
                raw=%raw_prefix,
                sanitized=%prefix,
                env=%frontend_service::METRICS_PREFIX_ENV,
                "Sanitized HTTP metrics prefix"
            );
        }
        let frontend_metric_name = |suffix: &str| format!("{}_{}", &prefix, suffix);

        let request_counter = IntCounterVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::REQUESTS_TOTAL),
                "Total number of LLM requests processed",
            ),
            &["model", "endpoint", "request_type", "status"],
        )
        .unwrap();

        let inflight_gauge = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::INFLIGHT_REQUESTS),
                "Number of inflight requests",
            ),
            &["model"],
        )
        .unwrap();

        let client_disconnect_gauge = prometheus::IntGauge::new(
            frontend_metric_name(frontend_service::DISCONNECTED_CLIENTS),
            "Number of disconnected clients",
        )
        .unwrap();

        let http_queue_gauge = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::QUEUED_REQUESTS),
                "Number of requests in HTTP processing queue",
            ),
            &["model"],
        )
        .unwrap();

        // Request duration buckets: configurable via DYN_METRICS_REQUEST_DURATION_{MIN,MAX,COUNT}
        let (req_dur_min, req_dur_max, req_dur_count) =
            parse_bucket_config("DYN_METRICS_REQUEST_DURATION", 1.0, 256.0, 10);
        let request_duration_buckets =
            generate_log_buckets(req_dur_min, req_dur_max, req_dur_count);

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::REQUEST_DURATION_SECONDS),
                "Duration of LLM requests",
            )
            .buckets(request_duration_buckets),
            &["model"],
        )
        .unwrap();

        // Input sequence length buckets: configurable via DYN_METRICS_INPUT_SEQUENCE_{MIN,MAX,COUNT}
        let (isl_min, isl_max, isl_count) =
            parse_bucket_config("DYN_METRICS_INPUT_SEQUENCE", 50.0, 128000.0, 12);
        let input_sequence_buckets = generate_log_buckets(isl_min, isl_max, isl_count);

        let input_sequence_length = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::INPUT_SEQUENCE_TOKENS),
                "Input sequence length in tokens",
            )
            .buckets(input_sequence_buckets.clone()),
            &["model"],
        )
        .unwrap();

        // Output sequence length buckets: configurable via DYN_METRICS_OUTPUT_SEQUENCE_{MIN,MAX,COUNT}
        let (osl_min, osl_max, osl_count) =
            parse_bucket_config("DYN_METRICS_OUTPUT_SEQUENCE", 50.0, 32000.0, 10);
        let output_sequence_buckets = generate_log_buckets(osl_min, osl_max, osl_count);

        let output_sequence_length = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::OUTPUT_SEQUENCE_TOKENS),
                "Output sequence length in tokens",
            )
            .buckets(output_sequence_buckets),
            &["model"],
        )
        .unwrap();

        let output_tokens_counter = IntCounterVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::OUTPUT_TOKENS_TOTAL),
                "Total number of output tokens generated (updates in real-time)",
            ),
            &["model"],
        )
        .unwrap();

        // Time to first token buckets: configurable via DYN_METRICS_TTFT_{MIN,MAX,COUNT}
        let (ttft_min, ttft_max, ttft_count) =
            parse_bucket_config("DYN_METRICS_TTFT", 0.001, 480.0, 18);
        let time_to_first_token_buckets = generate_log_buckets(ttft_min, ttft_max, ttft_count);

        let time_to_first_token = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::TIME_TO_FIRST_TOKEN_SECONDS),
                "Time to first token in seconds",
            )
            .buckets(time_to_first_token_buckets),
            &["model"],
        )
        .unwrap();

        // Inter-token latency buckets: configurable via DYN_METRICS_ITL_{MIN,MAX,COUNT}
        let (itl_min, itl_max, itl_count) = parse_bucket_config("DYN_METRICS_ITL", 0.001, 2.0, 13);
        let inter_token_latency_buckets = generate_log_buckets(itl_min, itl_max, itl_count);

        let inter_token_latency = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::INTER_TOKEN_LATENCY_SECONDS),
                "Inter-token latency in seconds",
            )
            .buckets(inter_token_latency_buckets),
            &["model"],
        )
        .unwrap();

        let cached_tokens = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::CACHED_TOKENS),
                "Number of cached tokens (prefix cache hits) per request",
            )
            .buckets(input_sequence_buckets.clone()),
            &["model"],
        )
        .unwrap();

        // Runtime configuration metrics
        // Note: Some of these metrics represent counter-like values from source systems,
        // but are implemented as gauges because they are copied/synchronized from upstream
        // counter values rather than being directly incremented.
        let model_total_kv_blocks = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_TOTAL_KV_BLOCKS),
                "Total KV cache blocks available for a worker serving the model",
            ),
            &["model"],
        )
        .unwrap();

        let model_max_num_seqs = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_MAX_NUM_SEQS),
                "Maximum number of sequences for a worker serving the model",
            ),
            &["model"],
        )
        .unwrap();

        let model_max_num_batched_tokens = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_MAX_NUM_BATCHED_TOKENS),
                "Maximum number of batched tokens for a worker serving the model",
            ),
            &["model"],
        )
        .unwrap();

        let model_context_length = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_CONTEXT_LENGTH),
                "Maximum context length in tokens for a worker serving the model",
            ),
            &["model"],
        )
        .unwrap();

        let model_kv_cache_block_size = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_KV_CACHE_BLOCK_SIZE),
                "KV cache block size in tokens for a worker serving the model",
            ),
            &["model"],
        )
        .unwrap();

        let model_migration_limit = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_MIGRATION_LIMIT),
                "Maximum number of request migrations allowed for the model",
            ),
            &["model"],
        )
        .unwrap();

        let model_migration_total = IntCounterVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_MIGRATION_TOTAL),
                "Total number of request migrations due to worker unavailability",
            ),
            &["model", frontend_service::MIGRATION_TYPE_LABEL],
        )
        .unwrap();

        Metrics {
            request_counter,
            inflight_gauge,
            client_disconnect_gauge,
            http_queue_gauge,
            request_duration,
            input_sequence_length,
            output_sequence_length,
            cached_tokens,
            output_tokens_counter,
            time_to_first_token,
            inter_token_latency,
            model_total_kv_blocks,
            model_max_num_seqs,
            model_max_num_batched_tokens,
            model_context_length,
            model_kv_cache_block_size,
            model_migration_limit,
            model_migration_total,
        }
    }

    /// Get the number of successful requests for the given dimensions:
    /// - model
    /// - endpoint (completions/chat_completions)
    /// - request type (unary/stream)
    /// - status (success/error)
    pub fn get_request_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
    ) -> u64 {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
            ])
            .get()
    }

    /// Increment the counter for requests for the given dimensions:
    /// - model
    /// - endpoint (completions/chat_completions)
    /// - request type (unary/stream)
    /// - status (success/error)
    fn inc_request_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
        status: &Status,
    ) {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
            ])
            .inc()
    }

    /// Get the number if inflight requests for the given model
    pub fn get_inflight_count(&self, model: &str) -> i64 {
        self.inflight_gauge.with_label_values(&[model]).get()
    }

    fn inc_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).inc()
    }

    fn dec_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).dec()
    }

    /// Increment the gauge for client disconnections
    pub fn inc_client_disconnect(&self) {
        self.client_disconnect_gauge.inc();
    }

    /// Get the count of client disconnections
    pub fn get_client_disconnect_count(&self) -> i64 {
        self.client_disconnect_gauge.get()
    }

    fn inc_http_queue_gauge(&self, model: &str) {
        self.http_queue_gauge.with_label_values(&[model]).inc()
    }

    fn dec_http_queue_gauge(&self, model: &str) {
        self.http_queue_gauge.with_label_values(&[model]).dec()
    }

    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.request_counter.clone()))?;
        registry.register(Box::new(self.inflight_gauge.clone()))?;
        registry.register(Box::new(self.client_disconnect_gauge.clone()))?;
        registry.register(Box::new(self.http_queue_gauge.clone()))?;
        registry.register(Box::new(self.request_duration.clone()))?;
        registry.register(Box::new(self.input_sequence_length.clone()))?;
        registry.register(Box::new(self.output_sequence_length.clone()))?;
        registry.register(Box::new(self.cached_tokens.clone()))?;
        registry.register(Box::new(self.output_tokens_counter.clone()))?;
        registry.register(Box::new(self.time_to_first_token.clone()))?;
        registry.register(Box::new(self.inter_token_latency.clone()))?;

        // Register runtime configuration metrics
        registry.register(Box::new(self.model_total_kv_blocks.clone()))?;
        registry.register(Box::new(self.model_max_num_seqs.clone()))?;
        registry.register(Box::new(self.model_max_num_batched_tokens.clone()))?;
        registry.register(Box::new(self.model_context_length.clone()))?;
        registry.register(Box::new(self.model_kv_cache_block_size.clone()))?;
        registry.register(Box::new(self.model_migration_limit.clone()))?;
        registry.register(Box::new(self.model_migration_total.clone()))?;

        Ok(())
    }

    /// Update runtime configuration metrics for a model
    /// This should be called when model runtime configuration is available or updated
    pub fn update_runtime_config_metrics(
        &self,
        model_name: &str,
        runtime_config: &ModelRuntimeConfig,
    ) {
        if let Some(total_kv_blocks) = runtime_config.total_kv_blocks {
            self.model_total_kv_blocks
                .with_label_values(&[model_name])
                .set(clamp_u64_to_i64(total_kv_blocks));
        }

        if let Some(max_num_seqs) = runtime_config.max_num_seqs {
            self.model_max_num_seqs
                .with_label_values(&[model_name])
                .set(clamp_u64_to_i64(max_num_seqs));
        }

        if let Some(max_batched_tokens) = runtime_config.max_num_batched_tokens {
            self.model_max_num_batched_tokens
                .with_label_values(&[model_name])
                .set(clamp_u64_to_i64(max_batched_tokens));
        }
    }

    /// Update metrics from a ModelDeploymentCard
    /// This updates both runtime config metrics and MDC-specific metrics
    pub fn update_metrics_from_mdc(&self, card: &ModelDeploymentCard) -> anyhow::Result<()> {
        self.update_runtime_config_metrics(&card.display_name, &card.runtime_config);

        self.model_context_length
            .with_label_values(&[&card.display_name])
            .set(card.context_length as i64);

        self.model_kv_cache_block_size
            .with_label_values(&[&card.display_name])
            .set(card.kv_cache_block_size as i64);

        self.model_migration_limit
            .with_label_values(&[&card.display_name])
            .set(card.migration_limit as i64);

        tracing::debug!(
            model = %card.display_name,
            "Successfully updated MDC metrics"
        );

        Ok(())
    }

    /// Increment the migration counter for a new request migration
    pub fn inc_migration_new_request(&self, model: &str) {
        self.model_migration_total
            .with_label_values(&[model, frontend_service::migration_type::NEW_REQUEST])
            .inc();
    }

    /// Increment the migration counter for an ongoing request migration
    pub fn inc_migration_ongoing_request(&self, model: &str) {
        self.model_migration_total
            .with_label_values(&[model, frontend_service::migration_type::ONGOING_REQUEST])
            .inc();
    }

    /// Get the current count of new request migrations for a model
    pub fn get_migration_new_request_count(&self, model: &str) -> u64 {
        self.model_migration_total
            .with_label_values(&[model, frontend_service::migration_type::NEW_REQUEST])
            .get()
    }

    /// Get the current count of ongoing request migrations for a model
    pub fn get_migration_ongoing_request_count(&self, model: &str) -> u64 {
        self.model_migration_total
            .with_label_values(&[model, frontend_service::migration_type::ONGOING_REQUEST])
            .get()
    }

    /// Create a new [`InflightGuard`] for the given model and annotate if its a streaming request,
    /// and the kind of endpoint that was hit
    ///
    /// The [`InflightGuard`] is an RAII object will handle incrementing the inflight gauge and
    /// request counters.
    ///
    /// # Metrics Distinction
    ///
    /// This method creates an inflight guard  t tracks requests actively being processed by the LLM engine.
    /// This is distinct from [`HttpQueueGuard`] which tracks requests from HTTP handler start until
    /// first token generation (including prefill time). The separation allows monitoring both HTTP processing queue time
    /// and actual LLM processing time.
    pub fn create_inflight_guard(
        self: Arc<Self>,
        model: &str,
        endpoint: Endpoint,
        streaming: bool,
    ) -> InflightGuard {
        let request_type = if streaming {
            RequestType::Stream
        } else {
            RequestType::Unary
        };

        InflightGuard::new(
            self.clone(),
            model.to_string().to_lowercase(),
            endpoint,
            request_type,
        )
    }

    /// Create a new [`ResponseMetricCollector`] for collecting per-response metrics (i.e., TTFT, ITL)
    pub fn create_response_collector(self: Arc<Self>, model: &str) -> ResponseMetricCollector {
        ResponseMetricCollector::new(self, model.to_string().to_lowercase())
    }

    /// Create a new [`HttpQueueGuard`] for tracking HTTP processing queue
    ///
    /// This guard tracks requests from HTTP handler start until first token generation,
    /// providing visibility into HTTP processing queue time before actual LLM processing begins.
    pub fn create_http_queue_guard(self: Arc<Self>, model: &str) -> HttpQueueGuard {
        HttpQueueGuard::new(self, model.to_string().to_lowercase())
    }
}

impl HttpQueueGuard {
    fn new(metrics: Arc<Metrics>, model: String) -> Self {
        // Increment the HTTP queue gauge when the guard is created
        metrics.inc_http_queue_gauge(&model);

        HttpQueueGuard { metrics, model }
    }
}

impl Drop for HttpQueueGuard {
    fn drop(&mut self) {
        // Decrement the HTTP queue gauge when the guard is dropped
        self.metrics.dec_http_queue_gauge(&self.model);
    }
}

impl InflightGuard {
    fn new(
        metrics: Arc<Metrics>,
        model: String,
        endpoint: Endpoint,
        request_type: RequestType,
    ) -> Self {
        // Start the timer
        let timer = Instant::now();

        // Increment the inflight gauge when the guard is created
        metrics.inc_inflight_gauge(&model);

        // Return the RAII Guard
        InflightGuard {
            metrics,
            model,
            endpoint,
            request_type,
            status: Status::Error,
            timer,
        }
    }

    pub(crate) fn mark_ok(&mut self) {
        self.status = Status::Success;
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let duration = self.timer.elapsed().as_secs_f64();

        // Decrement the gauge when the guard is dropped
        self.metrics.dec_inflight_gauge(&self.model);

        // the frequency on incrementing the full request counter is relatively low
        // if we were incrementing the counter on every forward pass, we'd use static CounterVec or
        // discrete counter object without the more costly lookup required for the following calls
        self.metrics.inc_request_counter(
            &self.model,
            &self.endpoint,
            &self.request_type,
            &self.status,
        );

        // Record the duration of the request
        self.metrics
            .request_duration
            .with_label_values(&[&self.model])
            .observe(duration);
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endpoint::Completions => write!(f, "completions"),
            Endpoint::ChatCompletions => write!(f, "chat_completions"),
            Endpoint::Embeddings => write!(f, "embeddings"),
            Endpoint::Responses => write!(f, "responses"),
            Endpoint::Tensor => write!(f, "tensor"),
        }
    }
}

impl Endpoint {
    pub fn as_str(&self) -> &'static str {
        match self {
            Endpoint::Completions => "completions",
            Endpoint::ChatCompletions => "chat_completions",
            Endpoint::Embeddings => "embeddings",
            Endpoint::Responses => "responses",
            Endpoint::Tensor => "tensor",
        }
    }
}

impl RequestType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestType::Unary => frontend_service::request_type::UNARY,
            RequestType::Stream => frontend_service::request_type::STREAM,
        }
    }
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Success => frontend_service::status::SUCCESS,
            Status::Error => frontend_service::status::ERROR,
        }
    }
}

impl ResponseMetricCollector {
    fn new(metrics: Arc<Metrics>, model: String) -> Self {
        ResponseMetricCollector {
            metrics,
            model,
            is_first_token: true,
            last_response_time: None,
            start_time: Instant::now(),
            osl: 0,
            cached_tokens_observed: false,
        }
    }

    /// Observe the current output sequence length
    pub fn observe_current_osl(&mut self, osl: usize) {
        self.osl = osl;
    }

    /// Check if this will be the first token (before calling observe_response)
    pub fn is_first_token(&self) -> bool {
        self.is_first_token
    }

    /// Observe cached tokens (prefix cache hits), observing only once per request when value is available
    pub fn observe_cached_tokens(&mut self, cached_tokens: Option<usize>) {
        if let Some(tokens) = cached_tokens
            && !self.cached_tokens_observed
        {
            self.cached_tokens_observed = true;
            self.metrics
                .cached_tokens
                .with_label_values(&[&self.model])
                .observe(tokens as f64);
        }
    }

    /// Observe a response with input sequence length and number of new tokens
    pub fn observe_response(&mut self, isl: usize, num_tokens: usize) {
        if num_tokens == 0 {
            return;
        }

        // Increment the real-time output tokens counter
        self.metrics
            .output_tokens_counter
            .with_label_values(&[&self.model])
            .inc_by(num_tokens as u64);

        if self.is_first_token {
            // NOTE: when there are multiple tokens in the first response,
            // we use the full response time as TTFT and ignore the ITL
            self.is_first_token = false;

            // Publish TTFT
            let ttft = self.start_time.elapsed().as_secs_f64();
            self.metrics
                .time_to_first_token
                .with_label_values(&[&self.model])
                .observe(ttft);

            // Publish ISL
            // TODO: publish ISL as soon as the tokenization process completes
            self.metrics
                .input_sequence_length
                .with_label_values(&[&self.model])
                .observe(isl as f64);
        }

        let current_duration = self.start_time.elapsed();

        if let Some(last_response_time) = self.last_response_time {
            let response_duration = current_duration - last_response_time;
            let itl = response_duration.as_secs_f64() / num_tokens as f64;
            for _ in 0..num_tokens {
                self.metrics
                    .inter_token_latency
                    .with_label_values(&[&self.model])
                    .observe(itl);
            }
        }

        self.last_response_time = Some(current_duration);
    }
}

impl Drop for ResponseMetricCollector {
    fn drop(&mut self) {
        // Publish final OSL when the collector is dropped
        self.metrics
            .output_sequence_length
            .with_label_values(&[&self.model])
            .observe(self.osl as f64);
    }
}

/// Process streaming metrics for annotated responses
///
/// This function handles metrics collection and http_queue_guard management for streaming responses.
/// It observes the current output sequence length, drops the http_queue_guard on the first token,
/// and records response metrics.
pub fn process_response_and_observe_metrics<T>(
    annotated: &crate::types::Annotated<T>,
    response_collector: &mut ResponseMetricCollector,
    http_queue_guard: &mut Option<HttpQueueGuard>,
) {
    use crate::preprocessor::LLMMetricAnnotation;

    // update metrics
    if let Ok(Some(metrics)) = LLMMetricAnnotation::from_annotation(annotated) {
        response_collector.observe_current_osl(metrics.output_tokens);

        // Drop http_queue_guard on first token for non-streaming (same as streaming)
        if response_collector.is_first_token()
            && metrics.chunk_tokens > 0
            && let Some(guard) = http_queue_guard.take()
        {
            drop(guard);
        }

        response_collector.observe_response(metrics.input_tokens, metrics.chunk_tokens);
    }
}

/// Event converter wrapper for streaming responses
pub struct EventConverter<T>(pub crate::types::Annotated<T>);

impl<T> From<crate::types::Annotated<T>> for EventConverter<T> {
    fn from(annotated: crate::types::Annotated<T>) -> Self {
        EventConverter(annotated)
    }
}

/// Process streaming response with event conversion for SSE
///
/// This function handles metrics collection, http_queue_guard management, and converts
/// annotated responses to SSE events for streaming responses.
///
/// Returns None for metrics annotation events (events without SSE data payload).
pub fn process_response_using_event_converter_and_observe_metrics<T: Serialize>(
    annotated: EventConverter<T>,
    response_collector: &mut ResponseMetricCollector,
    http_queue_guard: &mut Option<HttpQueueGuard>,
) -> Result<Option<Event>, axum::Error> {
    use crate::preprocessor::LLMMetricAnnotation;

    let mut annotated = annotated.0;

    // update metrics
    if let Ok(Some(metrics)) = LLMMetricAnnotation::from_annotation(&annotated) {
        response_collector.observe_current_osl(metrics.output_tokens);
        response_collector.observe_cached_tokens(metrics.cached_tokens);

        // Drop http_queue_guard on first token for streaming
        if response_collector.is_first_token()
            && metrics.chunk_tokens > 0
            && let Some(guard) = http_queue_guard.take()
        {
            drop(guard);
        }

        response_collector.observe_response(metrics.input_tokens, metrics.chunk_tokens);

        // Chomp the LLMMetricAnnotation so it's not returned in the response stream
        // TODO: add a flag to control what is returned in the SSE stream
        if annotated.event.as_deref() == Some(crate::preprocessor::ANNOTATION_LLM_METRICS) {
            annotated.event = None;
            annotated.comment = None;
        }
    }

    let mut event = Event::default();

    if let Some(ref data) = annotated.data {
        event = event.json_data(data)?;
    }

    if let Some(ref msg) = annotated.event {
        if msg == "error" {
            let msgs = annotated
                .comment
                .unwrap_or_else(|| vec!["unspecified error".to_string()]);
            return Err(axum::Error::new(msgs.join(" -- ")));
        }
        event = event.event(msg);
    }

    if let Some(comments) = annotated.comment {
        for comment in comments {
            event = event.comment(comment);
        }
    }

    // Filter out metrics annotation events (events without SSE data payload)
    if annotated.data.is_none() && annotated.event.is_none() {
        Ok(None)
    } else {
        Ok(Some(event))
    }
}

/// Create a new router with optional custom backend metrics support
pub fn router(registry: Registry, path: Option<String>) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| "/metrics".to_string());
    let doc = RouteDoc::new(axum::http::Method::GET, &path);

    let metrics_state = MetricsHandlerState {
        registry: Arc::new(registry),
    };

    let route = Router::new()
        .route(&path, get(handler_metrics))
        .with_state(Arc::new(metrics_state));
    (vec![doc], route)
}

/// Unified metrics handler
async fn handler_metrics(State(state): State<Arc<MetricsHandlerState>>) -> impl IntoResponse {
    // Gather and encode metrics
    // Note: If nim_on_demand is enabled, the NimMetricsCollector registered with the registry
    // will automatically call poll_nim_backend_stats when gather() is invoked
    let encoder = prometheus::TextEncoder::new();
    let metric_families = state.registry.gather();
    let mut buffer = vec![];
    if encoder.encode(&metric_families, &mut buffer).is_err() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode metrics",
        )
            .into_response();
    }

    let metrics = match String::from_utf8(buffer) {
        Ok(metrics) => metrics,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to encode metrics",
            )
                .into_response();
        }
    };

    (StatusCode::OK, metrics).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to_sig_figs() {
        // Test rounding to 2 significant figures
        assert_eq!(round_to_sig_figs(0.0026, 2), 0.0026);
        assert_eq!(round_to_sig_figs(0.26, 2), 0.26);
        assert_eq!(round_to_sig_figs(0.2356, 2), 0.24);
        assert_eq!(round_to_sig_figs(1.234, 2), 1.2);
        assert_eq!(round_to_sig_figs(12.34, 2), 12.0);
        assert_eq!(round_to_sig_figs(123.4, 2), 120.0);
        assert_eq!(round_to_sig_figs(1234.0, 2), 1200.0);
        assert_eq!(round_to_sig_figs(0.0, 2), 0.0);

        // Test edge cases
        assert_eq!(round_to_sig_figs(0.999, 2), 1.0);
        assert_eq!(round_to_sig_figs(9.99, 2), 10.0);
        assert_eq!(round_to_sig_figs(99.9, 2), 100.0);
    }

    #[test]
    fn test_generate_log_buckets_basic() {
        // Test basic properties
        let buckets = generate_log_buckets(1.0, 100.0, 5);

        // Check length
        assert_eq!(buckets.len(), 5);

        // Check first value is 0
        assert_eq!(buckets[0], 0.0);

        // Check last value is approximately max (rounded to 2 sig figs)
        assert_eq!(buckets[buckets.len() - 1], 100.0);

        // Check values are increasing
        for i in 1..buckets.len() {
            assert!(
                buckets[i] > buckets[i - 1],
                "Bucket values should be increasing: {} <= {}",
                buckets[i - 1],
                buckets[i]
            );
        }
    }

    #[test]
    fn test_generate_log_buckets_edge_cases() {
        // Test empty buckets
        let buckets = generate_log_buckets(1.0, 100.0, 0);
        assert_eq!(buckets.len(), 0);

        // Test single bucket
        let buckets = generate_log_buckets(1.0, 100.0, 1);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0], 0.0);

        // Test two buckets
        let buckets = generate_log_buckets(1.0, 100.0, 2);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0], 0.0);
        assert_eq!(buckets[1], 100.0);
    }

    #[test]
    fn test_generate_log_buckets_always_includes_zero() {
        // Test various configurations
        for count in 1..=20 {
            let buckets = generate_log_buckets(0.1, 1000.0, count);
            assert_eq!(
                buckets[0], 0.0,
                "First bucket should always be 0.0 for count={}",
                count
            );
        }
    }

    #[test]
    fn test_all_buckets_are_two_sig_figs() {
        let test_cases = vec![
            (1.0, 256.0, 10),
            (50.0, 128000.0, 12),
            (50.0, 32000.0, 10),
            (0.001, 480.0, 18),
            (0.001, 2.0, 13),
        ];

        for (min, max, count) in test_cases {
            let buckets = generate_log_buckets(min, max, count);
            for &value in buckets.iter().skip(1) {
                let rounded = round_to_sig_figs(value, 2);
                assert_eq!(
                    value, rounded,
                    "Value {} should be rounded to 2 sig figs (min={}, max={}, count={})",
                    value, min, max, count
                );
            }
        }
    }

    #[test]
    fn test_sig_fig_limitation_with_many_buckets() {
        // This test demonstrates that 2 sig figs limits the number of unique bucket values
        // With 1000 requested buckets but only 2 sig figs, we'll get automatic deduplication
        let buckets = generate_log_buckets(0.0001, 1.0, 1000);

        println!(
            "Requested 1000 buckets, got {} total values (including 0.0)",
            buckets.len()
        );

        // With 2 sig figs across 4 orders of magnitude (0.0001 to 1.0),
        // we can have roughly 90 unique values per order of magnitude
        // So we expect around 360 unique values maximum
        assert!(
            buckets.len() < 500,
            "Expected fewer than 500 unique buckets due to 2 sig fig limitation, got {}",
            buckets.len()
        );

        // Verify all values are unique (no duplicates remain after deduplication)
        let mut sorted_buckets = buckets.clone();
        sorted_buckets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_buckets.dedup();
        assert_eq!(
            buckets.len(),
            sorted_buckets.len(),
            "All buckets should be unique after deduplication"
        );

        // Verify first is still 0.0
        assert_eq!(buckets[0], 0.0);

        // Verify values are still in increasing order
        for i in 1..buckets.len() {
            assert!(
                buckets[i] > buckets[i - 1],
                "Buckets should be in increasing order"
            );
        }
    }

    #[test]
    fn test_deduplication_preserves_order() {
        // Test that deduplication maintains increasing order
        let buckets = generate_log_buckets(0.01, 1.0, 50);

        // Verify all values are unique
        let mut unique_check = std::collections::HashSet::new();
        for &bucket in &buckets {
            assert!(
                unique_check.insert(bucket.to_bits()),
                "Duplicate value {} found after deduplication",
                bucket
            );
        }

        // Verify order is maintained
        for i in 1..buckets.len() {
            assert!(
                buckets[i] > buckets[i - 1],
                "Bucket values should be in increasing order after deduplication"
            );
        }
    }

    #[test]
    fn test_output_tokens_counter_increments() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";

        // Create response collector
        let mut collector = metrics.clone().create_response_collector(model);

        // Simulate first chunk (5 tokens)
        collector.observe_response(100, 5);

        // Verify counter incremented by 5
        let counter_value = metrics
            .output_tokens_counter
            .with_label_values(&[model])
            .get();
        assert_eq!(counter_value, 5);

        // Simulate second chunk (10 tokens)
        collector.observe_response(100, 10);

        // Verify counter incremented to 15
        let counter_value = metrics
            .output_tokens_counter
            .with_label_values(&[model])
            .get();
        assert_eq!(counter_value, 15);

        // Simulate third chunk (7 tokens)
        collector.observe_response(100, 7);

        // Verify counter incremented to 22
        let counter_value = metrics
            .output_tokens_counter
            .with_label_values(&[model])
            .get();
        assert_eq!(counter_value, 22);
    }

    #[test]
    fn test_output_tokens_counter_zero_tokens() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let mut collector = metrics.clone().create_response_collector(model);

        // Simulate chunk with zero tokens (should not increment)
        collector.observe_response(100, 0);

        // Verify counter remains 0
        let counter_value = metrics
            .output_tokens_counter
            .with_label_values(&[model])
            .get();
        assert_eq!(counter_value, 0);

        // Add some tokens
        collector.observe_response(100, 5);
        assert_eq!(
            metrics
                .output_tokens_counter
                .with_label_values(&[model])
                .get(),
            5
        );

        // Try zero tokens again (should not change counter)
        collector.observe_response(100, 0);
        assert_eq!(
            metrics
                .output_tokens_counter
                .with_label_values(&[model])
                .get(),
            5
        );
    }

    #[test]
    fn test_output_tokens_counter_multiple_models() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model1 = "model-1";
        let model2 = "model-2";

        // Create collectors for different models
        let mut collector1 = metrics.clone().create_response_collector(model1);
        let mut collector2 = metrics.clone().create_response_collector(model2);

        // Increment model1
        collector1.observe_response(100, 10);
        assert_eq!(
            metrics
                .output_tokens_counter
                .with_label_values(&[model1])
                .get(),
            10
        );
        assert_eq!(
            metrics
                .output_tokens_counter
                .with_label_values(&[model2])
                .get(),
            0
        );

        // Increment model2
        collector2.observe_response(200, 20);
        assert_eq!(
            metrics
                .output_tokens_counter
                .with_label_values(&[model1])
                .get(),
            10
        );
        assert_eq!(
            metrics
                .output_tokens_counter
                .with_label_values(&[model2])
                .get(),
            20
        );

        // Increment model1 again
        collector1.observe_response(100, 5);
        assert_eq!(
            metrics
                .output_tokens_counter
                .with_label_values(&[model1])
                .get(),
            15
        );
        assert_eq!(
            metrics
                .output_tokens_counter
                .with_label_values(&[model2])
                .get(),
            20
        );
    }

    #[test]
    fn test_cached_tokens_once_per_request() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let expected_metric_name = "dynamo_frontend_cached_tokens";
        let mut collector = metrics.clone().create_response_collector(model);

        // Create histogram handle first
        let _histogram = metrics.cached_tokens.with_label_values(&[model]);

        // First call should observe and record 1 sample
        collector.observe_cached_tokens(Some(100));
        let metric_families = registry.gather();
        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_metric_name)
            .expect("histogram should be registered");
        assert_eq!(
            histogram_family.get_metric()[0]
                .get_histogram()
                .get_sample_count(),
            1
        );

        // Second call with same collector should not observe again (idempotent)
        collector.observe_cached_tokens(Some(50));
        let metric_families = registry.gather();
        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_metric_name)
            .expect("histogram should be registered");
        assert_eq!(
            histogram_family.get_metric()[0]
                .get_histogram()
                .get_sample_count(),
            1
        );

        // Third call with different value should still be idempotent
        collector.observe_cached_tokens(Some(75));
        let metric_families = registry.gather();
        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_metric_name)
            .expect("histogram should be registered");
        assert_eq!(
            histogram_family.get_metric()[0]
                .get_histogram()
                .get_sample_count(),
            1
        );
    }

    #[test]
    fn test_metrics_annotation_event_handling() {
        use crate::preprocessor::LLMMetricAnnotation;
        use crate::types::Annotated;

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let expected_metric_name = "dynamo_frontend_cached_tokens";
        let mut collector = metrics.clone().create_response_collector(model);

        // Create a metrics annotation event (event without SSE data payload)
        let mut annotated = Annotated::<
            crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse,
        > {
            id: None,
            data: None,
            event: Some(crate::preprocessor::ANNOTATION_LLM_METRICS.to_string()),
            comment: None,
        };

        // Add metrics annotation with cached_tokens
        let llm_metrics = LLMMetricAnnotation {
            input_tokens: 10,
            output_tokens: 20,
            chunk_tokens: 5,
            cached_tokens: Some(15),
        };

        let annotation = llm_metrics.to_annotation::<()>().unwrap();
        annotated.event = annotation.event;
        annotated.comment = annotation.comment;

        // Process the event
        let mut http_queue_guard = None;
        let result = process_response_using_event_converter_and_observe_metrics(
            EventConverter::from(annotated),
            &mut collector,
            &mut http_queue_guard,
        );

        // Should return Ok(None) for metrics annotation events
        assert!(matches!(result, Ok(None)));

        // Should have observed the cached tokens from the metrics annotation event
        let metric_families = registry.gather();
        let histogram_family = metric_families
            .iter()
            .find(|mf| mf.name() == expected_metric_name)
            .expect("histogram should be registered");
        assert_eq!(
            histogram_family.get_metric()[0]
                .get_histogram()
                .get_sample_count(),
            1
        );
    }
}
