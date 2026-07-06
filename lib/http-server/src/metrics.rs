// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use dynamo_runtime::config::environment_names::llm::metrics as env_metrics;
use dynamo_runtime::error::{ErrorType as DynamoErrorType, match_error_chain};
use dynamo_runtime::metrics::prometheus_names::{
    clamp_u64_to_i64, frontend_service, name_prefix, sanitize_frontend_prometheus_prefix,
};
pub use prometheus::Registry;
use prometheus::{GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts};

pub const WORKER_TYPE_PREFILL: &str = "prefill";
pub const WORKER_TYPE_DECODE: &str = "decode";
const UNSET_DP_RANK_LABEL: &str = "none";
const ITL_LOCAL_FLUSH_TOKENS: u64 = 64;

/// Check whether an error chain indicates the request was rejected.
pub fn request_was_rejected(err: &(dyn std::error::Error + 'static)) -> bool {
    match_error_chain(err, &[DynamoErrorType::ResourceExhausted], &[])
}

/// Check whether an error chain indicates that no backend worker is available.
pub fn request_was_unavailable(err: &(dyn std::error::Error + 'static)) -> bool {
    match_error_chain(err, &[DynamoErrorType::Unavailable], &[])
}

/// Check whether an error chain indicates the request was cancelled.
pub fn request_was_cancelled(err: &(dyn std::error::Error + 'static)) -> bool {
    match_error_chain(err, &[DynamoErrorType::Cancelled], &[])
}

/// Global Prometheus gauge for last observed TTFT per worker (in seconds)
/// Labels: worker_id, dp_rank, worker_type
pub static WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_LAST_TIME_TO_FIRST_TOKEN_SECONDS
            ),
            "Last observed time to first token per worker (seconds)",
        ),
        &["worker_id", "dp_rank", "worker_type"],
    )
    .expect("Failed to create worker_last_time_to_first_token gauge")
});

/// Global Prometheus gauge for last observed input sequence tokens per worker
/// Labels: worker_id, dp_rank, worker_type
/// Updated atomically with TTFT - represents the input token count from the same request
pub static WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    IntGaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_LAST_INPUT_SEQUENCE_TOKENS
            ),
            "Last observed input sequence tokens per worker",
        ),
        &["worker_id", "dp_rank", "worker_type"],
    )
    .expect("Failed to create worker_last_input_sequence_tokens gauge")
});

/// Global Prometheus gauge for last observed ITL per worker (in seconds)
/// Labels: worker_id, dp_rank, worker_type
pub static WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            format!(
                "{}_{}",
                name_prefix::FRONTEND,
                frontend_service::WORKER_LAST_INTER_TOKEN_LATENCY_SECONDS
            ),
            "Last observed inter-token latency per worker (seconds)",
        ),
        &["worker_id", "dp_rank", "worker_type"],
    )
    .expect("Failed to create worker_last_inter_token_latency gauge")
});

/// Register the global per-worker TTFT/ITL/input-tokens Prometheus metrics with the given registry.
///
/// This should be called once during HTTP service setup to expose the metrics
/// via the `/metrics` endpoint.
///
/// # Errors
/// Returns an error if the metrics are already registered with the registry.
pub fn register_worker_timing_metrics(registry: &Registry) -> Result<(), prometheus::Error> {
    registry.register(Box::new(WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE.clone()))?;
    registry.register(Box::new(WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE.clone()))?;
    registry.register(Box::new(WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE.clone()))?;
    Ok(())
}

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
pub fn generate_log_buckets(min: f64, max: f64, count: usize) -> Vec<f64> {
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
pub fn round_to_sig_figs(value: f64, sig_figs: u32) -> f64 {
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

pub struct Metrics {
    request_started_counter: IntCounterVec,
    request_counter: IntCounterVec,
    /// Deprecated: use `active_requests_gauge`. Kept for backwards compatibility.
    inflight_gauge: IntGaugeVec,
    active_requests_gauge: IntGaugeVec,
    client_disconnect_gauge: prometheus::IntGauge,
    http_queue_gauge: IntGaugeVec,
    request_duration: HistogramVec,
    input_sequence_length: HistogramVec,
    output_sequence_length: HistogramVec,
    cached_tokens: HistogramVec,
    tokenizer_latency: HistogramVec,
    output_tokens_counter: IntCounterVec,
    time_to_first_token: HistogramVec,
    inter_token_latency: HistogramVec,
    /// End-to-end latency of an OpenAI `/v1/embeddings` request. Distinct from
    /// `request_duration` so the buckets can be sized for sub-second pooling
    /// inference rather than the 1-512s LLM-generation range. Labeled by
    /// `model`.
    embedding_latency: HistogramVec,

    // Per-request multimodal content-part count histograms (labeled by `model`).
    images_per_request: HistogramVec,
    videos_per_request: HistogramVec,
    audio_per_request: HistogramVec,

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
    model_migration_max_seq_len_exceeded_total: IntCounterVec,
    model_cancellation_total: IntCounterVec,
    model_rejection_total: IntCounterVec,
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
    error_type: ErrorType,
    timer: Instant,
    request_id: String,
    span: tracing::Span,
}

/// Requests will be logged by the type of endpoint hit
/// This will include llamastack in the future
#[derive(Clone, Copy)]
pub enum Endpoint {
    /// OAI Completions
    Completions,

    /// OAI Chat Completions
    ChatCompletions,

    /// OAI Embeddings
    Embeddings,

    /// OAI Images
    Images,

    /// OAI Videos
    Videos,

    /// OAI Audio Speech
    Audios,

    /// OAI Responses
    Responses,

    /// Anthropic Messages
    AnthropicMessages,

    /// Tensor
    Tensor,

    /// Generate (token-in/token-out)
    Generate,
}

/// Metrics for the HTTP service
pub enum RequestType {
    /// SingleIn / SingleOut
    Unary,

    /// SingleIn / ManyOut
    Stream,
}

/// Labels for cancellation metrics
pub struct CancellationLabels {
    pub model: String,
    pub endpoint: String,
    pub request_type: String,
}

/// Status
#[derive(PartialEq)]
pub enum Status {
    Success,
    Error,
}

/// Error type classification for fine-grained observability
#[derive(PartialEq, Clone, Debug)]
pub enum ErrorType {
    /// No error (for successful requests)
    None,
    /// Client validation error (4xx with "Validation:" prefix)
    Validation,
    /// Model or resource not found (404)
    NotFound,
    /// Service overloaded or rate limited (429 or 529)
    Overload,
    /// Service unavailable because no backend worker can serve the request
    Unavailable,
    /// Request cancelled by client or timeout
    Cancelled,
    /// Backend accepted the request but stopped responding (response inactivity timeout)
    ResponseTimeout,
    /// Internal server error (500 and other unexpected errors)
    Internal,
    /// Feature not implemented (501)
    NotImplemented,
}

/// Track response-specific metrics
pub struct ResponseMetricCollector {
    metrics: Arc<Metrics>,
    model: String,
    // Per-model metric handles cached for the request. Most are resolved at construction;
    // ITL is resolved lazily on its first observation so requests that never produce ITL
    // do not allocate a local histogram. Caching avoids re-hashing the `model` label on
    // every chunk or output token. Each handle shares the underlying metric with its vec,
    // so observations are equivalent.
    output_tokens_counter: prometheus::IntCounter,
    time_to_first_token: prometheus::Histogram,
    inter_token_latency: Option<prometheus::local::LocalHistogram>,
    itl_pending_tokens: u64,
    input_sequence_length: prometheus::Histogram,
    cached_tokens: prometheus::Histogram,
    // Per-model multimodal count histogram handles, resolved once at construction.
    images_per_request: prometheus::Histogram,
    videos_per_request: prometheus::Histogram,
    audio_per_request: prometheus::Histogram,
    // Latched per-request counts (for the tracing span fields recorded in `Drop`).
    image_count_val: usize,
    video_count_val: usize,
    audio_count_val: usize,
    multimodal_counts_observed: bool,
    start_time: Instant,
    // we use is_first_token to distinguish TTFT from ITL. It is true by default and
    // flipped to false when the first token is returned and TTFT is published.
    is_first_token: bool,
    // we track the last response time so that ITL for the newly returned tokens can
    // be computed.
    last_response_time: Option<Duration>,
    osl: usize,
    isl: usize,
    ttft_ms: Option<f64>,
    itl_sum_secs: f64,
    itl_count: u64,
    // we track if cached_tokens has been observed to ensure we only increment once per request
    cached_tokens_observed: bool,
    // we track if tokenize latency has been observed to ensure we only increment once per request
    tokenize_latency_observed: bool,
    // latest accumulated detokenize latency and sample count reported by tracker
    detokenize_latency_total: Duration,
    detokenize_count_total: u64,
    // Prefill worker info for TTFT attribution (set from request metrics)
    prefill_worker_id: Option<u64>,
    prefill_dp_rank: Option<u32>,
    // Prefill worker type for Prometheus labeling - stored at routing time to avoid MDC lookup
    prefill_worker_type: Option<String>,
    // Decode worker info for ITL attribution (set from request metrics)
    decode_worker_id: Option<u64>,
    decode_dp_rank: Option<u32>,
    // Decode worker type for Prometheus labeling - stored at routing time to avoid MDC lookup
    decode_worker_type: Option<String>,
    // Cached per-worker ITL gauge handle. The decode-worker labels are latched once at
    // routing time (`set_worker_info` only sets when unset), so this GaugeVec handle is
    // resolved a single time and reused — the per-token observe path then does no label
    // formatting (`worker_id.to_string()`) or label hashing, only `set`.
    decode_itl_gauge: Option<prometheus::Gauge>,
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
    /// - `{prefix}_tokenizer_latency_ms` - HistogramVec for tokenizer latency in milliseconds
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
    /// - `DYN_METRICS_EMBEDDING_LATENCY_{MIN,MAX,COUNT}` - End-to-end `/v1/embeddings` latency histogram (defaults: 0.001, 10.0, 14)
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
        Self::new_with_prefix(std::env::var(frontend_service::METRICS_PREFIX_ENV).ok())
    }

    /// Create Metrics with an explicit optional prefix. `None` uses the standard
    /// frontend prefix and does not read environment variables.
    pub fn new_with_prefix(metrics_prefix: Option<String>) -> Self {
        // TODO: Remove DYN_METRICS_PREFIX env-var override (added in PR #2432 for
        // NIM compatibility with the old "nv_llm_http_service_" prefix). No longer
        // needed — hardcode name_prefix::FRONTEND and drop the sanitize function.
        let raw_prefix = metrics_prefix.unwrap_or_else(|| name_prefix::FRONTEND.to_string());
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
            &["model", "endpoint", "request_type", "status", "error_type"],
        )
        .unwrap();

        let request_started_counter = IntCounterVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::REQUESTS_STARTED_TOTAL),
                "Total number of LLM requests accepted by the frontend handler",
            ),
            &["model", "endpoint", "request_type"],
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

        let active_requests_gauge = IntGaugeVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::ACTIVE_REQUESTS),
                "Number of requests currently being handled by the frontend, from HTTP handler entry to response completion",
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
            parse_bucket_config("DYN_METRICS_REQUEST_DURATION", 1.0, 512.0, 10);
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

        // Embedding latency buckets: pooling-model inference is typically
        // sub-second (60-200 token inputs on L4-class GPUs land in the 5-50ms
        // range), so 1ms..10s on a log scale gives p50/p99 resolution that
        // the 1..512s `request_duration` buckets cannot.
        let (emb_min, emb_max, emb_count) =
            parse_bucket_config("DYN_METRICS_EMBEDDING_LATENCY", 0.001, 10.0, 14);
        let embedding_latency_buckets = generate_log_buckets(emb_min, emb_max, emb_count);

        let embedding_latency = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::EMBEDDING_LATENCY_SECONDS),
                "End-to-end latency of /v1/embeddings requests, in seconds. \
                 Distinct from request_duration_seconds so the bucket range can \
                 cover sub-second pooling-model inference.",
            )
            .buckets(embedding_latency_buckets),
            &["model"],
        )
        .unwrap();

        // Per-request media-count buckets: every integer through 10 (where most
        // requests land), then coarse steps for the large-batch tail.
        let multimodal_count_buckets = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0,
        ];

        let images_per_request = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::IMAGES_PER_REQUEST),
                "Number of image_url content parts per request",
            )
            .buckets(multimodal_count_buckets.clone()),
            &["model"],
        )
        .unwrap();

        let videos_per_request = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::VIDEOS_PER_REQUEST),
                "Number of video_url content parts per request",
            )
            .buckets(multimodal_count_buckets.clone()),
            &["model"],
        )
        .unwrap();

        let audio_per_request = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::AUDIO_PER_REQUEST),
                "Number of audio_url content parts per request",
            )
            .buckets(multimodal_count_buckets),
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

        let tokenizer_latency = HistogramVec::new(
            HistogramOpts::new(
                frontend_metric_name(frontend_service::TOKENIZER_LATENCY_MS),
                "Tokenizer latency in milliseconds",
            )
            .buckets(vec![
                0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
            ]),
            &[frontend_service::OPERATION_LABEL],
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

        let model_migration_max_seq_len_exceeded_total = IntCounterVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_MIGRATION_MAX_SEQ_LEN_EXCEEDED_TOTAL),
                "Total number of times migration was disabled by max_seq_len limit",
            ),
            &["model"],
        )
        .unwrap();

        let model_cancellation_total = IntCounterVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_CANCELLATION_TOTAL),
                "Total number of request cancellations",
            ),
            &["model", "endpoint", "request_type"],
        )
        .unwrap();

        let model_rejection_total = IntCounterVec::new(
            Opts::new(
                frontend_metric_name(frontend_service::MODEL_REJECTION_TOTAL),
                "Total number of requests rejected due to resource exhaustion",
            ),
            &["model", "endpoint"],
        )
        .unwrap();

        Metrics {
            request_started_counter,
            request_counter,
            inflight_gauge,
            active_requests_gauge,
            client_disconnect_gauge,
            http_queue_gauge,
            request_duration,
            input_sequence_length,
            output_sequence_length,
            cached_tokens,
            tokenizer_latency,
            output_tokens_counter,
            time_to_first_token,
            inter_token_latency,
            embedding_latency,
            images_per_request,
            videos_per_request,
            audio_per_request,
            model_total_kv_blocks,
            model_max_num_seqs,
            model_max_num_batched_tokens,
            model_context_length,
            model_kv_cache_block_size,
            model_migration_limit,
            model_migration_total,
            model_migration_max_seq_len_exceeded_total,
            model_cancellation_total,
            model_rejection_total,
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
        error_type: &ErrorType,
    ) -> u64 {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
                error_type.as_str(),
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
        error_type: &ErrorType,
    ) {
        self.request_counter
            .with_label_values(&[
                model,
                endpoint.as_str(),
                request_type.as_str(),
                status.as_str(),
                error_type.as_str(),
            ])
            .inc()
    }

    /// Increment the counter for requests accepted by the frontend handler.
    fn inc_request_started_counter(
        &self,
        model: &str,
        endpoint: &Endpoint,
        request_type: &RequestType,
    ) {
        self.request_started_counter
            .with_label_values(&[model, endpoint.as_str(), request_type.as_str()])
            .inc()
    }

    /// Get the number if inflight requests for the given model
    pub fn get_inflight_count(&self, model: &str) -> i64 {
        self.inflight_gauge.with_label_values(&[model]).get()
    }

    /// Record one observation of the end-to-end latency of an
    /// `/v1/embeddings` request, in seconds. Labeled by model.
    ///
    /// Negative values are dropped (they can only happen from a clock skew or
    /// programming bug; recording them would corrupt percentile estimates).
    pub fn observe_embedding_latency(&self, model: &str, seconds: f64) {
        if !seconds.is_finite() || seconds < 0.0 {
            return;
        }
        self.embedding_latency
            .with_label_values(&[model])
            .observe(seconds);
    }

    /// Get the cumulative count of embedding latency observations for the given
    /// model. Test helper.
    #[cfg(test)]
    pub fn embedding_latency_count(&self, model: &str) -> u64 {
        self.embedding_latency
            .with_label_values(&[model])
            .get_sample_count()
    }

    /// Get the cumulative sum (seconds) of embedding latency observations for
    /// the given model. Test helper.
    #[cfg(test)]
    pub fn embedding_latency_sum(&self, model: &str) -> f64 {
        self.embedding_latency
            .with_label_values(&[model])
            .get_sample_sum()
    }

    fn inc_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).inc();
        self.active_requests_gauge.with_label_values(&[model]).inc();
    }

    fn dec_inflight_gauge(&self, model: &str) {
        self.inflight_gauge.with_label_values(&[model]).dec();
        self.active_requests_gauge.with_label_values(&[model]).dec();
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
        registry.register(Box::new(self.request_started_counter.clone()))?;
        registry.register(Box::new(self.request_counter.clone()))?;
        registry.register(Box::new(self.inflight_gauge.clone()))?;
        registry.register(Box::new(self.active_requests_gauge.clone()))?;
        registry.register(Box::new(self.client_disconnect_gauge.clone()))?;
        registry.register(Box::new(self.http_queue_gauge.clone()))?;
        registry.register(Box::new(self.request_duration.clone()))?;
        registry.register(Box::new(self.input_sequence_length.clone()))?;
        registry.register(Box::new(self.output_sequence_length.clone()))?;
        registry.register(Box::new(self.cached_tokens.clone()))?;
        registry.register(Box::new(self.tokenizer_latency.clone()))?;
        registry.register(Box::new(self.output_tokens_counter.clone()))?;
        registry.register(Box::new(self.time_to_first_token.clone()))?;
        registry.register(Box::new(self.inter_token_latency.clone()))?;
        registry.register(Box::new(self.embedding_latency.clone()))?;
        registry.register(Box::new(self.images_per_request.clone()))?;
        registry.register(Box::new(self.videos_per_request.clone()))?;
        registry.register(Box::new(self.audio_per_request.clone()))?;

        // Register runtime configuration metrics
        registry.register(Box::new(self.model_total_kv_blocks.clone()))?;
        registry.register(Box::new(self.model_max_num_seqs.clone()))?;
        registry.register(Box::new(self.model_max_num_batched_tokens.clone()))?;
        registry.register(Box::new(self.model_context_length.clone()))?;
        registry.register(Box::new(self.model_kv_cache_block_size.clone()))?;
        registry.register(Box::new(self.model_migration_limit.clone()))?;
        registry.register(Box::new(self.model_migration_total.clone()))?;
        registry.register(Box::new(
            self.model_migration_max_seq_len_exceeded_total.clone(),
        ))?;
        registry.register(Box::new(self.model_cancellation_total.clone()))?;
        registry.register(Box::new(self.model_rejection_total.clone()))?;

        Ok(())
    }

    /// Update model runtime-capacity gauges from scalar values.
    pub fn update_runtime_config_metrics_raw(
        &self,
        model_name: &str,
        total_kv_blocks: Option<u64>,
        max_num_seqs: Option<u64>,
        max_num_batched_tokens: Option<u64>,
    ) {
        if let Some(value) = total_kv_blocks {
            self.model_total_kv_blocks
                .with_label_values(&[model_name])
                .set(clamp_u64_to_i64(value));
        }
        if let Some(value) = max_num_seqs {
            self.model_max_num_seqs
                .with_label_values(&[model_name])
                .set(clamp_u64_to_i64(value));
        }
        if let Some(value) = max_num_batched_tokens {
            self.model_max_num_batched_tokens
                .with_label_values(&[model_name])
                .set(clamp_u64_to_i64(value));
        }
    }

    /// Update model deployment gauges from scalar values.
    pub fn update_deployment_metrics(
        &self,
        model_name: &str,
        context_length: u32,
        kv_cache_block_size: u32,
        migration_limit: u32,
    ) {
        self.model_context_length
            .with_label_values(&[model_name])
            .set(context_length as i64);
        self.model_kv_cache_block_size
            .with_label_values(&[model_name])
            .set(kv_cache_block_size as i64);
        self.model_migration_limit
            .with_label_values(&[model_name])
            .set(migration_limit as i64);
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

    /// Increment the counter for migrations disabled by max_seq_len being exceeded
    pub fn inc_migration_max_seq_len_exceeded(&self, model: &str) {
        self.model_migration_max_seq_len_exceeded_total
            .with_label_values(&[model])
            .inc();
    }

    /// Get the current count of migrations disabled by max_seq_len being exceeded
    pub fn get_migration_max_seq_len_exceeded_count(&self, model: &str) -> u64 {
        self.model_migration_max_seq_len_exceeded_total
            .with_label_values(&[model])
            .get()
    }

    /// Increment the cancellation counter
    pub fn inc_cancellation(&self, labels: &CancellationLabels) {
        self.model_cancellation_total
            .with_label_values(&[&labels.model, &labels.endpoint, &labels.request_type])
            .inc();
    }

    /// Get the current cancellation count
    pub fn get_cancellation_count(&self, labels: &CancellationLabels) -> u64 {
        self.model_cancellation_total
            .with_label_values(&[&labels.model, &labels.endpoint, &labels.request_type])
            .get()
    }

    /// Increment the rejection counter for a request rejected due to resource exhaustion
    pub fn inc_rejection(&self, model: &str, endpoint: Endpoint) {
        self.model_rejection_total
            .with_label_values(&[model, &endpoint.to_string()])
            .inc();
    }

    /// Get the current rejection count for a model and endpoint
    pub fn get_rejection_count(&self, model: &str, endpoint: Endpoint) -> u64 {
        self.model_rejection_total
            .with_label_values(&[model, &endpoint.to_string()])
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
        request_id: &str,
    ) -> InflightGuard {
        let request_type = if streaming {
            RequestType::Stream
        } else {
            RequestType::Unary
        };

        InflightGuard::new(
            self.clone(),
            model.to_string(),
            endpoint,
            request_type,
            request_id.to_string(),
        )
    }

    /// Create a new [`ResponseMetricCollector`] for collecting per-response metrics (i.e., TTFT, ITL)
    pub fn create_response_collector(self: Arc<Self>, model: &str) -> ResponseMetricCollector {
        ResponseMetricCollector::new(self, model.to_string())
    }

    /// Create a new [`HttpQueueGuard`] for tracking HTTP processing queue
    ///
    /// This guard tracks requests from HTTP handler start until first token generation,
    /// providing visibility into HTTP processing queue time before actual LLM processing begins.
    pub fn create_http_queue_guard(self: Arc<Self>, model: &str) -> HttpQueueGuard {
        HttpQueueGuard::new(self, model.to_string())
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
        request_id: String,
    ) -> Self {
        let timer = Instant::now();
        metrics.inc_inflight_gauge(&model);
        metrics.inc_request_started_counter(&model, &endpoint, &request_type);

        tracing::Span::current().record("model", model.as_str());

        tracing::info!(
            request_id = %request_id,
            model = %model,
            endpoint = %endpoint,
            request_type = %request_type,
            "request received"
        );

        InflightGuard {
            metrics,
            model,
            endpoint,
            request_type,
            status: Status::Error,
            error_type: ErrorType::Internal,
            timer,
            request_id,
            span: tracing::Span::current(),
        }
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }
    pub fn request_type(&self) -> &RequestType {
        &self.request_type
    }
    pub fn error_type(&self) -> &ErrorType {
        &self.error_type
    }
    pub fn elapsed_ms(&self) -> u128 {
        self.timer.elapsed().as_millis()
    }

    pub fn mark_ok(&mut self) {
        self.status = Status::Success;
        self.error_type = ErrorType::None;
    }

    pub fn mark_error(&mut self, error_type: ErrorType) {
        self.status = Status::Error;
        self.error_type = error_type;
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let _enter = self.span.enter();
        let duration = self.timer.elapsed().as_secs_f64();
        self.metrics.dec_inflight_gauge(&self.model);
        self.metrics.inc_request_counter(
            &self.model,
            &self.endpoint,
            &self.request_type,
            &self.status,
            &self.error_type,
        );
        self.metrics
            .request_duration
            .with_label_values(&[&self.model])
            .observe(duration);

        let elapsed_ms = (duration * 1000.0) as u64;
        let status_str = self.status.as_str();
        match self.status {
            Status::Error => {
                let detail = match self.error_type {
                    ErrorType::Cancelled => "cancelled before completion",
                    ErrorType::ResponseTimeout => "backend stream inactivity timeout",
                    ErrorType::Internal => "internal server error during processing",
                    ErrorType::Validation => "invalid request parameters",
                    ErrorType::NotFound => "model or resource not found",
                    ErrorType::Overload => "service overloaded or rate limited",
                    ErrorType::Unavailable => "no backend worker available",
                    ErrorType::NotImplemented => "requested feature not implemented",
                    ErrorType::None => "unknown error",
                };
                tracing::error!(
                    request_id = %self.request_id,
                    model = %self.model,
                    endpoint = %self.endpoint,
                    request_type = %self.request_type,
                    status = %status_str,
                    error_type = %self.error_type,
                    error_detail = %detail,
                    elapsed_ms = %elapsed_ms,
                    "request completed"
                );
            }
            Status::Success => {
                tracing::info!(
                    request_id = %self.request_id,
                    model = %self.model,
                    endpoint = %self.endpoint,
                    request_type = %self.request_type,
                    status = %status_str,
                    elapsed_ms = %elapsed_ms,
                    "request completed"
                );
            }
        }
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Endpoint::Completions => write!(f, "completions"),
            Endpoint::ChatCompletions => write!(f, "chat_completions"),
            Endpoint::Embeddings => write!(f, "embeddings"),
            Endpoint::Images => write!(f, "images"),
            Endpoint::Videos => write!(f, "videos"),
            Endpoint::Audios => write!(f, "audios"),
            Endpoint::Responses => write!(f, "responses"),
            Endpoint::AnthropicMessages => write!(f, "anthropic_messages"),
            Endpoint::Tensor => write!(f, "tensor"),
            Endpoint::Generate => write!(f, "generate"),
        }
    }
}

impl Endpoint {
    pub fn as_str(&self) -> &'static str {
        match self {
            Endpoint::Completions => "completions",
            Endpoint::ChatCompletions => "chat_completions",
            Endpoint::Embeddings => "embeddings",
            Endpoint::Images => "images",
            Endpoint::Videos => "videos",
            Endpoint::Audios => "audios",
            Endpoint::Responses => "responses",
            Endpoint::AnthropicMessages => "anthropic_messages",
            Endpoint::Tensor => "tensor",
            Endpoint::Generate => "generate",
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

impl std::fmt::Display for RequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
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

impl ErrorType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorType::None => frontend_service::error_type::NONE,
            ErrorType::Validation => frontend_service::error_type::VALIDATION,
            ErrorType::NotFound => frontend_service::error_type::NOT_FOUND,
            ErrorType::Overload => frontend_service::error_type::OVERLOAD,
            ErrorType::Unavailable => frontend_service::error_type::UNAVAILABLE,
            ErrorType::Cancelled => frontend_service::error_type::CANCELLED,
            ErrorType::ResponseTimeout => frontend_service::error_type::RESPONSE_TIMEOUT,
            ErrorType::Internal => frontend_service::error_type::INTERNAL,
            ErrorType::NotImplemented => frontend_service::error_type::NOT_IMPLEMENTED,
        }
    }
}

impl std::fmt::Display for ErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ResponseMetricCollector {
    fn new(metrics: Arc<Metrics>, model: String) -> Self {
        // Resolve the per-model handles once (cheap clones of the vec entries) so the
        // per-chunk / per-token hot path in `observe_response` does no label hashing.
        let output_tokens_counter = metrics.output_tokens_counter.with_label_values(&[&model]);
        let time_to_first_token = metrics.time_to_first_token.with_label_values(&[&model]);
        let input_sequence_length = metrics.input_sequence_length.with_label_values(&[&model]);
        let cached_tokens = metrics.cached_tokens.with_label_values(&[&model]);
        let images_per_request = metrics.images_per_request.with_label_values(&[&model]);
        let videos_per_request = metrics.videos_per_request.with_label_values(&[&model]);
        let audio_per_request = metrics.audio_per_request.with_label_values(&[&model]);
        ResponseMetricCollector {
            metrics,
            model,
            output_tokens_counter,
            time_to_first_token,
            inter_token_latency: None,
            itl_pending_tokens: 0,
            input_sequence_length,
            cached_tokens,
            images_per_request,
            videos_per_request,
            audio_per_request,
            image_count_val: 0,
            video_count_val: 0,
            audio_count_val: 0,
            multimodal_counts_observed: false,
            is_first_token: true,
            last_response_time: None,
            start_time: Instant::now(),
            osl: 0,
            isl: 0,
            ttft_ms: None,
            itl_sum_secs: 0.0,
            itl_count: 0,
            cached_tokens_observed: false,
            tokenize_latency_observed: false,
            detokenize_latency_total: Duration::ZERO,
            detokenize_count_total: 0,
            prefill_worker_id: None,
            prefill_dp_rank: None,
            prefill_worker_type: None,
            decode_worker_id: None,
            decode_dp_rank: None,
            decode_worker_type: None,
            decode_itl_gauge: None,
        }
    }

    /// Set the worker info for per-worker TTFT/ITL metrics.
    /// In disaggregated mode, TTFT is attributed to prefill worker, ITL to decode worker.
    /// Worker types are stored at routing time to avoid expensive MDC lookup when updating metrics.
    pub fn set_worker_info(
        &mut self,
        prefill_worker_id: Option<u64>,
        prefill_dp_rank: Option<u32>,
        prefill_worker_type: Option<&str>,
        decode_worker_id: Option<u64>,
        decode_dp_rank: Option<u32>,
        decode_worker_type: Option<&str>,
    ) {
        if self.prefill_worker_id.is_none() {
            self.prefill_worker_id = prefill_worker_id;
        }
        if self.prefill_dp_rank.is_none() {
            self.prefill_dp_rank = prefill_dp_rank;
        }
        if self.prefill_worker_type.is_none() {
            self.prefill_worker_type = prefill_worker_type.map(str::to_owned);
        }
        if self.decode_worker_id.is_none() {
            self.decode_worker_id = decode_worker_id;
        }
        if self.decode_dp_rank.is_none() {
            self.decode_dp_rank = decode_dp_rank;
        }
        if self.decode_worker_type.is_none() {
            self.decode_worker_type = decode_worker_type.map(str::to_owned);
        }
    }

    fn local_inter_token_latency(&mut self) -> &mut prometheus::local::LocalHistogram {
        let metrics = &self.metrics;
        let model = self.model.as_str();
        self.inter_token_latency.get_or_insert_with(|| {
            metrics
                .inter_token_latency
                .with_label_values(&[model])
                .local()
        })
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
            self.cached_tokens.observe(tokens as f64);
        }
    }

    /// Observe per-request multimodal content-part counts, latched to run exactly
    /// once per request (the counts are constant across a request's chunks).
    pub fn observe_multimodal_counts(
        &mut self,
        image_count: usize,
        video_count: usize,
        audio_count: usize,
    ) {
        if self.multimodal_counts_observed {
            return;
        }
        self.multimodal_counts_observed = true;

        self.image_count_val = image_count;
        self.video_count_val = video_count;
        self.audio_count_val = audio_count;

        self.images_per_request.observe(image_count as f64);
        self.videos_per_request.observe(video_count as f64);
        self.audio_per_request.observe(audio_count as f64);
    }

    /// Observe tokenize/detokenize latencies in milliseconds.
    /// Tokenize is observed once per request; detokenize is accumulated and observed at request end.
    pub fn observe_tokenize_latencies(
        &mut self,
        tokenize_latency: Option<Duration>,
        detokenize_latency: Option<Duration>,
        detokenize_count: Option<u64>,
    ) {
        if let Some(latency) = tokenize_latency
            && !self.tokenize_latency_observed
        {
            self.tokenize_latency_observed = true;
            self.metrics
                .tokenizer_latency
                .with_label_values(&[frontend_service::operation::TOKENIZE])
                .observe(latency.as_secs_f64() * 1000.0);
        }

        if let Some(latency) = detokenize_latency {
            self.detokenize_latency_total = latency;
        }
        if let Some(count) = detokenize_count {
            self.detokenize_count_total = count;
        }
    }

    /// Observe a response with input sequence length and number of new tokens
    pub fn observe_response(&mut self, isl: usize, num_tokens: usize) {
        if num_tokens == 0 {
            return;
        }

        // Store ISL for span recording on drop
        self.isl = isl;

        // Increment the real-time output tokens counter
        self.output_tokens_counter.inc_by(num_tokens as u64);

        if self.is_first_token {
            // NOTE: when there are multiple tokens in the first response,
            // we use the full response time as TTFT and ignore the ITL
            self.is_first_token = false;

            // Publish TTFT and store for span recording
            let ttft = self.start_time.elapsed().as_secs_f64();
            self.ttft_ms = Some(ttft * 1000.0);
            self.time_to_first_token.observe(ttft);

            // Update per-worker TTFT and input sequence tokens gauges - attributed to prefill worker.
            // Both gauges are updated atomically from the same request to correlate latency with input size.
            // Use stored worker_type (from routing time) to avoid MDC lookup.
            // Falls back to WORKER_TYPE_PREFILL if not available.
            if let Some(worker_id) = self.prefill_worker_id {
                let worker_id_str = worker_id.to_string();
                let dp_rank_str = self
                    .prefill_dp_rank
                    .map_or(UNSET_DP_RANK_LABEL.to_string(), |r| r.to_string());
                let worker_type = self
                    .prefill_worker_type
                    .as_deref()
                    .unwrap_or(WORKER_TYPE_PREFILL);
                let labels = &[worker_id_str.as_str(), dp_rank_str.as_str(), worker_type];
                WORKER_LAST_TIME_TO_FIRST_TOKEN_GAUGE
                    .with_label_values(labels)
                    .set(ttft);
                WORKER_LAST_INPUT_SEQUENCE_TOKENS_GAUGE
                    .with_label_values(labels)
                    .set(isl as i64);
            }

            // Publish ISL
            // TODO: publish ISL as soon as the tokenization process completes
            self.input_sequence_length.observe(isl as f64);
        }

        let current_duration = self.start_time.elapsed();

        if let Some(last_response_time) = self.last_response_time {
            let response_duration = current_duration - last_response_time;
            let itl = response_duration.as_secs_f64() / num_tokens as f64;
            self.itl_sum_secs += itl * num_tokens as f64;
            self.itl_count += num_tokens as u64;
            self.itl_pending_tokens = self.itl_pending_tokens.saturating_add(num_tokens as u64);
            let should_flush = self.itl_pending_tokens >= ITL_LOCAL_FLUSH_TOKENS;
            {
                // Resolve the request-local histogram on the first ITL only, then reuse
                // it without re-hashing the model label for each output token.
                let histogram = self.local_inter_token_latency();
                for _ in 0..num_tokens {
                    histogram.observe(itl);
                }
                if should_flush {
                    histogram.flush();
                }
            }
            if should_flush {
                self.itl_pending_tokens = 0;
            }

            // Update per-worker ITL gauge - attributed to decode worker.
            // Use stored worker_type (from routing time) to avoid MDC lookup.
            // Falls back to WORKER_TYPE_DECODE if not available.
            //
            // The worker labels are fixed for the request's lifetime (latched in
            // `set_worker_info`), so resolve the GaugeVec handle once and cache it.
            // This keeps the per-token path free of `to_string()` allocations and
            // label hashing; subsequent tokens only call `set`.
            if self.decode_itl_gauge.is_none()
                && let Some(worker_id) = self.decode_worker_id
            {
                let worker_id_str = worker_id.to_string();
                let dp_rank_str = self
                    .decode_dp_rank
                    .map_or(UNSET_DP_RANK_LABEL.to_string(), |r| r.to_string());
                let worker_type = self
                    .decode_worker_type
                    .as_deref()
                    .unwrap_or(WORKER_TYPE_DECODE);
                self.decode_itl_gauge =
                    Some(WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE.with_label_values(&[
                        worker_id_str.as_str(),
                        dp_rank_str.as_str(),
                        worker_type,
                    ]));
            }
            if let Some(gauge) = &self.decode_itl_gauge {
                gauge.set(itl);
            }
        }

        self.last_response_time = Some(current_duration);
    }
}

impl Drop for ResponseMetricCollector {
    fn drop(&mut self) {
        if let Some(histogram) = &self.inter_token_latency {
            histogram.flush();
        }
        self.itl_pending_tokens = 0;

        if !self.detokenize_latency_total.is_zero() && self.detokenize_count_total > 0 {
            let avg_detokenize_latency_ms = (self.detokenize_latency_total.as_secs_f64() * 1000.0)
                / self.detokenize_count_total as f64;
            self.metrics
                .tokenizer_latency
                .with_label_values(&[frontend_service::operation::DETOKENIZE])
                .observe(avg_detokenize_latency_ms);
        }

        // Publish final OSL when the collector is dropped, but only for
        // requests that actually produced output tokens. Recording zero for
        // failed/cancelled requests would corrupt histogram averages (sum/count)
        // and percentiles. Failures are already tracked by requests_total with
        // status and error_type labels.
        if self.osl > 0 {
            self.metrics
                .output_sequence_length
                .with_label_values(&[&self.model])
                .observe(self.osl as f64);
        }

        // Record request summary on the enclosing span.
        // InflightGuard::Drop and on_response logs will inherit these.
        let span = tracing::Span::current();
        span.record("input_tokens", self.isl as u32);
        span.record("output_tokens", self.osl as u32);
        // Only record once observed, so requests that never carried a metrics
        // annotation (e.g. errored before the first chunk) don't stamp a false zero.
        if self.multimodal_counts_observed {
            span.record("image_count", self.image_count_val as u32);
            span.record("video_count", self.video_count_val as u32);
            span.record("audio_count", self.audio_count_val as u32);
        }
        if let Some(ttft_ms) = self.ttft_ms {
            span.record("ttft_ms", format!("{:.2}", ttft_ms).as_str());
        }
        if self.itl_count > 0 {
            let avg_ms = (self.itl_sum_secs / self.itl_count as f64) * 1000.0;
            span.record("avg_itl_ms", format!("{:.2}", avg_ms).as_str());
        }
        if let Some(worker_id) = self.prefill_worker_id {
            span.record("prefill_worker_id", worker_id);
        }
        if let Some(worker_id) = self.decode_worker_id {
            span.record("decode_worker_id", worker_id);
        }
    }
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
    fn test_local_itl_histogram_is_initialized_lazily() {
        let metrics = Arc::new(Metrics::new());
        let model = "lazy-local-itl-model";
        let global = metrics.inter_token_latency.with_label_values(&[model]);
        let mut collector = metrics.create_response_collector(model);

        assert!(collector.inter_token_latency.is_none());

        collector.observe_response(10, 0);
        assert!(
            collector.inter_token_latency.is_none(),
            "zero-token responses must not allocate a local histogram"
        );

        collector.observe_response(10, 1);
        assert!(
            collector.inter_token_latency.is_none(),
            "the TTFT-only response must not allocate a local histogram"
        );

        collector.observe_response(10, 1);
        assert!(
            collector.inter_token_latency.is_some(),
            "the first ITL observation must allocate a local histogram"
        );
        assert_eq!(
            global.get_sample_count(),
            0,
            "the first ITL observation must remain local until flush or drop"
        );

        drop(collector);
        assert_eq!(global.get_sample_count(), 1);
    }

    #[test]
    fn test_cached_handles_record_through_vec() {
        // The collector caches per-model handles; observing through them must update
        // the same metric the vec exposes. Regression guard for the handle cache
        // (including the lazily resolved handle used by the per-token ITL loop).
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        let model = "cached-handle-model";

        let mut collector = metrics.clone().create_response_collector(model);
        // First chunk (3 tokens): TTFT + ISL observed once; no ITL yet.
        collector.observe_response(42, 3);
        // Second chunk (4 tokens): ITL observed once per token via the cached handle.
        collector.observe_response(42, 4);

        // ITL observations are request-local until the batch threshold or drop.
        assert_eq!(
            metrics
                .inter_token_latency
                .with_label_values(&[model])
                .get_sample_count(),
            0
        );
        drop(collector);

        let ttft = metrics.time_to_first_token.with_label_values(&[model]);
        assert_eq!(ttft.get_sample_count(), 1, "TTFT observed once");

        let isl = metrics.input_sequence_length.with_label_values(&[model]);
        assert_eq!(isl.get_sample_count(), 1, "ISL observed once");
        assert_eq!(isl.get_sample_sum(), 42.0);

        let itl = metrics.inter_token_latency.with_label_values(&[model]);
        assert_eq!(
            itl.get_sample_count(),
            4,
            "ITL observed once per token of 2nd chunk"
        );

        let out = metrics
            .output_tokens_counter
            .with_label_values(&[model])
            .get();
        assert_eq!(out, 7, "output tokens = 3 + 4 via cached counter handle");
    }

    #[test]
    fn test_local_itl_histogram_flushes_at_threshold_and_drop() {
        use prometheus::core::Collector;

        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        let model = "local-itl-flush-model";
        let global = metrics.inter_token_latency.with_label_values(&[model]);

        let mut collector = metrics.clone().create_response_collector(model);
        collector.observe_response(10, 1); // TTFT only.
        assert!(collector.inter_token_latency.is_none());
        collector.observe_response(10, 63);
        assert!(collector.inter_token_latency.is_some());
        assert_eq!(
            global.get_sample_count(),
            0,
            "a partial local batch must remain unpublished"
        );

        collector.observe_response(10, 1);
        assert_eq!(
            global.get_sample_count(),
            ITL_LOCAL_FLUSH_TOKENS,
            "the 64th pending ITL sample must flush the local histogram"
        );

        collector.observe_response(10, 7);
        assert_eq!(
            global.get_sample_count(),
            ITL_LOCAL_FLUSH_TOKENS,
            "the next partial batch must remain local"
        );
        drop(collector);

        assert_eq!(global.get_sample_count(), 71);
        assert!(global.get_sample_sum() > 0.0);

        let families = global.collect();
        let histogram = families[0].get_metric()[0].get_histogram();
        let last_bucket = histogram.get_bucket().last().unwrap();
        assert_eq!(
            last_bucket.cumulative_count(),
            global.get_sample_count(),
            "flushing must update histogram buckets as well as count and sum"
        );
    }

    #[test]
    fn test_worker_metadata_is_latched_without_replacement() {
        let metrics = Arc::new(Metrics::new());
        let mut collector = metrics.create_response_collector("worker-latch-model");
        collector.set_worker_info(
            Some(11),
            Some(1),
            Some("prefill-first"),
            Some(22),
            Some(2),
            Some("decode-first"),
        );
        collector.set_worker_info(
            Some(33),
            Some(3),
            Some("prefill-later"),
            Some(44),
            Some(4),
            Some("decode-later"),
        );

        assert_eq!(collector.prefill_worker_id, Some(11));
        assert_eq!(collector.prefill_dp_rank, Some(1));
        assert_eq!(
            collector.prefill_worker_type.as_deref(),
            Some("prefill-first")
        );
        assert_eq!(collector.decode_worker_id, Some(22));
        assert_eq!(collector.decode_dp_rank, Some(2));
        assert_eq!(
            collector.decode_worker_type.as_deref(),
            Some("decode-first")
        );
    }

    #[test]
    fn test_multimodal_counts_observed_once() {
        // Counts are request-level constants carried on every chunk's annotation.
        // The collector must latch them on the first observation: each histogram
        // records exactly one per-request sample, and later (differing) values are
        // ignored. The histogram `_sum` doubles as the cumulative volume.
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        let model = "mm-counts-model";

        let mut collector = metrics.clone().create_response_collector(model);
        // Repeated calls simulate the same counts arriving on each streamed chunk.
        collector.observe_multimodal_counts(2, 1, 0);
        collector.observe_multimodal_counts(2, 1, 0);
        // A later, differing value must not override the latched counts.
        collector.observe_multimodal_counts(99, 99, 99);

        let img = metrics.images_per_request.with_label_values(&[model]);
        assert_eq!(img.get_sample_count(), 1, "image histogram observed once");
        assert_eq!(img.get_sample_sum(), 2.0, "image _sum == cumulative volume");
        let vid = metrics.videos_per_request.with_label_values(&[model]);
        assert_eq!(vid.get_sample_count(), 1, "video histogram observed once");
        assert_eq!(vid.get_sample_sum(), 1.0);
        let aud = metrics.audio_per_request.with_label_values(&[model]);
        assert_eq!(
            aud.get_sample_count(),
            1,
            "audio histogram observes even a 0 (text-only distribution)"
        );
        assert_eq!(aud.get_sample_sum(), 0.0);
    }

    #[test]
    fn test_multimodal_counts_text_only_zeros() {
        // A text-only request carries zero counts; the histograms still get one
        // sample each (at 0), so `_count` is 1 and `_sum` is 0.
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();
        let model = "text-only-model";

        let mut collector = metrics.clone().create_response_collector(model);
        collector.observe_multimodal_counts(0, 0, 0);

        let img = metrics.images_per_request.with_label_values(&[model]);
        assert_eq!(img.get_sample_count(), 1);
        assert_eq!(img.get_sample_sum(), 0.0);
    }

    #[test]
    fn test_cached_decode_itl_gauge_records_for_worker() {
        // The per-worker ITL gauge handle is resolved once (worker labels are latched
        // at routing time via `set_worker_info`) and reused on every output token.
        // Verify observations through the cached handle land on the same GaugeVec
        // series the labels address. Regression guard for the per-token handle cache.
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).ok();
        let model = "itl-gauge-model";

        // Distinctive worker id so this never collides with other tests on the
        // process-global WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE.
        let worker_id: u64 = 9_876_543_210;
        let mut collector = metrics.clone().create_response_collector(model);
        collector.set_worker_info(None, None, None, Some(worker_id), Some(0), Some("decode"));

        // First chunk: TTFT only, no ITL yet (no prior response time) -> gauge unset.
        collector.observe_response(10, 1);
        // Elapse measurable time so ITL > 0, then a second chunk drives the cached
        // per-worker ITL gauge.
        std::thread::sleep(std::time::Duration::from_millis(2));
        collector.observe_response(10, 1);

        let worker_id_str = worker_id.to_string();
        let gauge = WORKER_LAST_INTER_TOKEN_LATENCY_GAUGE.with_label_values(&[
            worker_id_str.as_str(),
            "0",
            "decode",
        ]);
        assert!(
            gauge.get() > 0.0,
            "cached per-worker ITL gauge must record a positive ITL through the vec, got {}",
            gauge.get()
        );
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
    fn test_error_type_as_str() {
        assert_eq!(ErrorType::None.as_str(), "");
        assert_eq!(ErrorType::Validation.as_str(), "validation");
        assert_eq!(ErrorType::NotFound.as_str(), "not_found");
        assert_eq!(ErrorType::Overload.as_str(), "overload");
        assert_eq!(ErrorType::Unavailable.as_str(), "unavailable");
        assert_eq!(ErrorType::Cancelled.as_str(), "cancelled");
        assert_eq!(ErrorType::ResponseTimeout.as_str(), "response_timeout");
        assert_eq!(ErrorType::Internal.as_str(), "internal");
        assert_eq!(ErrorType::NotImplemented.as_str(), "not_implemented");
    }

    #[test]
    fn test_inflight_guard_marks_success_with_correct_error_type() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";

        {
            let mut guard =
                metrics
                    .clone()
                    .create_inflight_guard(model, Endpoint::ChatCompletions, false, "");
            guard.mark_ok();
        } // guard drops here

        // Verify counter incremented with status=success, error_type=""
        let counter_value = metrics
            .request_counter
            .with_label_values(&[
                model,
                Endpoint::ChatCompletions.as_str(),
                RequestType::Unary.as_str(),
                Status::Success.as_str(),
                ErrorType::None.as_str(),
            ])
            .get();
        assert_eq!(counter_value, 1);

        let started_counter_value = metrics
            .request_started_counter
            .with_label_values(&[
                model,
                Endpoint::ChatCompletions.as_str(),
                RequestType::Unary.as_str(),
            ])
            .get();
        assert_eq!(started_counter_value, 1);
    }

    #[test]
    fn test_inflight_guard_marks_validation_error() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";

        {
            let mut guard =
                metrics
                    .clone()
                    .create_inflight_guard(model, Endpoint::ChatCompletions, false, "");
            guard.mark_error(ErrorType::Validation);
        } // guard drops here

        // Verify counter incremented with status=error, error_type=validation
        let counter_value = metrics
            .request_counter
            .with_label_values(&[
                model,
                Endpoint::ChatCompletions.as_str(),
                RequestType::Unary.as_str(),
                Status::Error.as_str(),
                ErrorType::Validation.as_str(),
            ])
            .get();
        assert_eq!(counter_value, 1);
    }

    #[test]
    fn test_inflight_guard_defaults_to_internal_error_on_drop() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";

        {
            let _guard =
                metrics
                    .clone()
                    .create_inflight_guard(model, Endpoint::ChatCompletions, false, "");
            // Don't call mark_ok() or mark_error() - simulate panic/unhandled error
        } // guard drops with default error_type=Internal

        // Verify counter incremented with status=error, error_type=internal
        let counter_value = metrics
            .request_counter
            .with_label_values(&[
                model,
                Endpoint::ChatCompletions.as_str(),
                RequestType::Unary.as_str(),
                Status::Error.as_str(),
                ErrorType::Internal.as_str(),
            ])
            .get();
        assert_eq!(counter_value, 1);
    }

    #[test]
    fn test_active_requests_tracks_inflight_guard() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model-active";

        // Both gauges start at 0
        assert_eq!(metrics.inflight_gauge.with_label_values(&[model]).get(), 0);
        assert_eq!(
            metrics
                .active_requests_gauge
                .with_label_values(&[model])
                .get(),
            0
        );

        {
            let _guard = metrics.clone().create_inflight_guard(
                model,
                Endpoint::ChatCompletions,
                true,
                "req-1",
            );

            // Both gauges increment together
            assert_eq!(metrics.inflight_gauge.with_label_values(&[model]).get(), 1);
            assert_eq!(
                metrics
                    .active_requests_gauge
                    .with_label_values(&[model])
                    .get(),
                1
            );

            {
                let _guard2 = metrics.clone().create_inflight_guard(
                    model,
                    Endpoint::ChatCompletions,
                    true,
                    "req-2",
                );
                assert_eq!(metrics.inflight_gauge.with_label_values(&[model]).get(), 2);
                assert_eq!(
                    metrics
                        .active_requests_gauge
                        .with_label_values(&[model])
                        .get(),
                    2
                );
            }
            // guard2 dropped
            assert_eq!(metrics.inflight_gauge.with_label_values(&[model]).get(), 1);
            assert_eq!(
                metrics
                    .active_requests_gauge
                    .with_label_values(&[model])
                    .get(),
                1
            );
        }
        // guard dropped — both back to 0
        assert_eq!(metrics.inflight_gauge.with_label_values(&[model]).get(), 0);
        assert_eq!(
            metrics
                .active_requests_gauge
                .with_label_values(&[model])
                .get(),
            0
        );
    }

    #[test]
    fn test_lifecycle_constructors_preserve_model_label_casing() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "Llama-3.1-8B-Instruct";

        let _inflight = metrics.clone().create_inflight_guard(
            model,
            Endpoint::ChatCompletions,
            true,
            "req-case",
        );
        let mut collector = metrics.clone().create_response_collector(model);
        collector.observe_response(100, 1);
        let _queue = metrics.clone().create_http_queue_guard(model);

        // Drop the lifecycle guards so their Drop impls record
        // completion-time metrics (request_counter, request_duration,
        // detokenize observations) before we gather. Without this the
        // regression only covers constructor-time labels.
        drop(_inflight);
        drop(collector);
        drop(_queue);

        let metric_families = registry.gather();
        let observed: Vec<String> = metric_families
            .iter()
            .flat_map(|mf| mf.get_metric())
            .flat_map(|m| m.get_label())
            .filter(|l| l.name() == "model")
            .map(|l| l.value().to_string())
            .collect();

        assert!(
            !observed.is_empty(),
            "expected at least one metric to carry a model label"
        );
        for value in &observed {
            assert_eq!(
                value, model,
                "model label was modified; expected original casing to be preserved"
            );
        }
    }

    #[test]
    fn test_all_error_types_recorded_correctly() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";
        let endpoint = Endpoint::ChatCompletions;

        // Test each error type
        let error_types = vec![
            ErrorType::Validation,
            ErrorType::NotFound,
            ErrorType::Overload,
            ErrorType::Unavailable,
            ErrorType::Cancelled,
            ErrorType::ResponseTimeout,
            ErrorType::Internal,
            ErrorType::NotImplemented,
        ];

        for error_type in &error_types {
            let mut guard = metrics
                .clone()
                .create_inflight_guard(model, endpoint, false, "");
            guard.mark_error(error_type.clone());
            drop(guard);
        }

        // Verify each error type recorded correctly
        for error_type in &error_types {
            let counter_value = metrics
                .request_counter
                .with_label_values(&[
                    model,
                    endpoint.as_str(),
                    RequestType::Unary.as_str(),
                    Status::Error.as_str(),
                    error_type.as_str(),
                ])
                .get();
            assert_eq!(
                counter_value,
                1,
                "Should have 1 request for error_type={}",
                error_type.as_str()
            );
        }
    }

    #[test]
    fn test_multiple_requests_different_error_types() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "test-model";

        // Record 2 validation errors, 3 internal errors, 1 success
        for _ in 0..2 {
            let mut guard =
                metrics
                    .clone()
                    .create_inflight_guard(model, Endpoint::ChatCompletions, false, "");
            guard.mark_error(ErrorType::Validation);
            drop(guard);
        }

        for _ in 0..3 {
            let mut guard =
                metrics
                    .clone()
                    .create_inflight_guard(model, Endpoint::Completions, false, "");
            guard.mark_error(ErrorType::Internal);
            drop(guard);
        }

        {
            let mut guard =
                metrics
                    .clone()
                    .create_inflight_guard(model, Endpoint::Embeddings, false, "");
            guard.mark_ok();
            drop(guard);
        }

        // Check validation errors (2 from ChatCompletions)
        let validation_count = metrics
            .request_counter
            .with_label_values(&[
                model,
                Endpoint::ChatCompletions.as_str(),
                RequestType::Unary.as_str(),
                Status::Error.as_str(),
                ErrorType::Validation.as_str(),
            ])
            .get();
        assert_eq!(validation_count, 2);

        // Check internal errors (3 from Completions)
        let internal_count = metrics
            .request_counter
            .with_label_values(&[
                model,
                Endpoint::Completions.as_str(),
                RequestType::Unary.as_str(),
                Status::Error.as_str(),
                ErrorType::Internal.as_str(),
            ])
            .get();
        assert_eq!(internal_count, 3);

        // Check success (1 from Embeddings)
        let success_count = metrics
            .request_counter
            .with_label_values(&[
                model,
                Endpoint::Embeddings.as_str(),
                RequestType::Unary.as_str(),
                Status::Success.as_str(),
                ErrorType::None.as_str(),
            ])
            .get();
        assert_eq!(success_count, 1);
    }

    #[test]
    fn test_observe_embedding_latency_records_observation() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "embed-test";
        assert_eq!(metrics.embedding_latency_count(model), 0);
        assert_eq!(metrics.embedding_latency_sum(model), 0.0);

        metrics.observe_embedding_latency(model, 0.005);
        metrics.observe_embedding_latency(model, 0.020);

        assert_eq!(metrics.embedding_latency_count(model), 2);
        assert!((metrics.embedding_latency_sum(model) - 0.025).abs() < 1e-9);
    }

    #[test]
    fn test_observe_embedding_latency_drops_invalid_values() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let model = "embed-test";

        metrics.observe_embedding_latency(model, -0.5);
        metrics.observe_embedding_latency(model, f64::NAN);
        metrics.observe_embedding_latency(model, f64::INFINITY);
        metrics.observe_embedding_latency(model, f64::NEG_INFINITY);

        assert_eq!(metrics.embedding_latency_count(model), 0);
        assert_eq!(metrics.embedding_latency_sum(model), 0.0);

        // Zero is allowed (boundary case for finite, non-negative).
        metrics.observe_embedding_latency(model, 0.0);
        assert_eq!(metrics.embedding_latency_count(model), 1);
    }

    #[test]
    fn test_observe_embedding_latency_partitions_by_model() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        let m1 = "embed-a";
        let m2 = "embed-b";

        metrics.observe_embedding_latency(m1, 0.010);
        metrics.observe_embedding_latency(m1, 0.030);
        metrics.observe_embedding_latency(m2, 0.005);

        assert_eq!(metrics.embedding_latency_count(m1), 2);
        assert_eq!(metrics.embedding_latency_count(m2), 1);
        assert!((metrics.embedding_latency_sum(m1) - 0.040).abs() < 1e-9);
        assert!((metrics.embedding_latency_sum(m2) - 0.005).abs() < 1e-9);
    }

    #[test]
    fn test_embedding_latency_registered_in_registry() {
        let metrics = Arc::new(Metrics::new());
        let registry = prometheus::Registry::new();
        metrics.register(&registry).unwrap();

        metrics.observe_embedding_latency("embed-test", 0.012);

        let metric_families = registry.gather();
        let found = metric_families.iter().any(|mf| {
            mf.name()
                .ends_with(frontend_service::EMBEDDING_LATENCY_SECONDS)
        });
        assert!(
            found,
            "embedding_latency_seconds histogram must be registered with the registry"
        );
    }
}
