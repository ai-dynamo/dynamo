// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo Distributed Logging Module.
//!
//! - Configuration loaded from:
//!   1. Environment variables (highest priority).
//!   2. Optional TOML file pointed to by the `DYN_LOGGING_CONFIG_PATH` environment variable.
//!   3. `/opt/dynamo/etc/logging.toml`.
//!
//! Logging can take two forms: `READABLE` or `JSONL`. The default is `READABLE`. `JSONL`
//! can be enabled by setting the `DYN_LOGGING_JSONL` environment variable to `1`.
//!
//! To use local timezone for logging timestamps, set the `DYN_LOG_USE_LOCAL_TZ` environment variable to `1`.
//!
//! Filters can be configured using the `DYN_LOG` environment variable or by setting the `filters`
//! key in the TOML configuration file. Filters are comma-separated key-value pairs where the key
//! is the crate or module name and the value is the log level. The default log level is `info`.
//!
//! Example:
//! ```toml
//! log_level = "error"
//!
//! [log_filters]
//! "test_logging" = "info"
//! "test_logging::api" = "trace"
//! ```

use std::collections::{BTreeMap, HashMap};
use std::sync::Once;

use figment::{
    Figment,
    providers::{Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use tracing::level_filters::LevelFilter;
use tracing::{Event, Subscriber};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::time::LocalTime;
use tracing_subscriber::fmt::time::SystemTime;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::fmt::{FmtContext, FormatFields};
use tracing_subscriber::fmt::{FormattedFields, format::Writer};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{filter::Directive, fmt};

use crate::config::{
    disable_ansi_logging, env_is_truthy, jsonl_logging_enabled, span_events_enabled,
};
use async_nats::{HeaderMap, HeaderValue};
use axum::extract::FromRequestParts;
use axum::http;
use axum::http::Request;
use axum::http::request::Parts;
use serde_json::Value;
use std::convert::Infallible;
use std::time::Instant;
use tower_http::trace::{DefaultMakeSpan, TraceLayer};
use tracing::Id;
use tracing::Span;
use tracing::field::Field;
use tracing::span;
use tracing_subscriber::Layer;
use tracing_subscriber::Registry;
use tracing_subscriber::field::Visit;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::SpanData;
use uuid::Uuid;

use opentelemetry::propagation::{Extractor, Injector, TextMapPropagator};
use opentelemetry::trace::{Span as OtelSpan, TraceContextExt};
use opentelemetry::{global, trace::Tracer};
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_otlp::WithExportConfig;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{Key, KeyValue};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_sdk::trace::Sampler;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing::error;
use tracing_subscriber::layer::SubscriberExt;
// use tracing_subscriber::Registry;

use std::time::Duration;
use tracing::{info, instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::config::environment_names::logging as env_logging;

/// Default log level
const DEFAULT_FILTER_LEVEL: &str = "info";

/// Default OTLP endpoint
const DEFAULT_OTLP_ENDPOINT: &str = "http://localhost:4317";

/// Default OTLP HTTP endpoint
const DEFAULT_OTLP_HTTP_ENDPOINT: &str = "http://localhost:4318";

/// Default service name
const DEFAULT_OTEL_SERVICE_NAME: &str = "dynamo";

/// Once instance to ensure the logger is only initialized once
static INIT: Once = Once::new();

#[derive(Serialize, Deserialize, Debug)]
struct LoggingConfig {
    log_level: String,
    log_filters: HashMap<String, String>,
}
impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            log_level: DEFAULT_FILTER_LEVEL.to_string(),
            log_filters: HashMap::from([
                ("h2".to_string(), "error".to_string()),
                ("tower".to_string(), "error".to_string()),
                ("hyper_util".to_string(), "error".to_string()),
                ("neli".to_string(), "error".to_string()),
                ("async_nats".to_string(), "error".to_string()),
                ("rustls".to_string(), "error".to_string()),
                ("tokenizers".to_string(), "error".to_string()),
                ("axum".to_string(), "error".to_string()),
                ("tonic".to_string(), "error".to_string()),
                ("hf_hub".to_string(), "error".to_string()),
                ("opentelemetry".to_string(), "error".to_string()),
                ("opentelemetry-otlp".to_string(), "error".to_string()),
                ("opentelemetry_sdk".to_string(), "error".to_string()),
            ]),
        }
    }
}

/// Check if OTLP trace exporting is enabled (accepts: "1", "true", "on", "yes" - case insensitive)
fn otlp_exporter_enabled() -> bool {
    env_is_truthy(env_logging::otlp::OTEL_EXPORT_ENABLED)
}

/// Get the service name from environment or use default
fn get_service_name() -> String {
    std::env::var(env_logging::otlp::OTEL_SERVICE_NAME)
        .unwrap_or_else(|_| DEFAULT_OTEL_SERVICE_NAME.to_string())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OtlpProtocol {
    Grpc,
    HttpProtobuf,
}

impl OtlpProtocol {
    fn as_str(self) -> &'static str {
        match self {
            Self::Grpc => "grpc",
            Self::HttpProtobuf => "http/protobuf",
        }
    }
}

fn parse_otlp_protocol_for_env(value: Option<&str>, env_name: &str) -> OtlpProtocol {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None => OtlpProtocol::Grpc,
        Some(value) if value.eq_ignore_ascii_case("grpc") => OtlpProtocol::Grpc,
        Some(value) if value.eq_ignore_ascii_case("http/protobuf") => OtlpProtocol::HttpProtobuf,
        Some(value) => {
            eprintln!(
                "WARNING: unsupported {} '{}'; falling back to grpc",
                env_name, value
            );
            OtlpProtocol::Grpc
        }
    }
}

fn parse_otlp_protocol(value: Option<&str>) -> OtlpProtocol {
    parse_otlp_protocol_for_env(value, env_logging::otlp::OTEL_EXPORTER_OTLP_PROTOCOL)
}

fn otlp_protocol_from_env() -> OtlpProtocol {
    parse_otlp_protocol(
        std::env::var(env_logging::otlp::OTEL_EXPORTER_OTLP_PROTOCOL)
            .ok()
            .as_deref(),
    )
}

fn resolve_signal_otlp_protocol(
    generic_protocol: OtlpProtocol,
    signal_protocol: Option<&str>,
    signal_protocol_env: &str,
) -> OtlpProtocol {
    match signal_protocol
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(value) => parse_otlp_protocol_for_env(Some(value), signal_protocol_env),
        None => generic_protocol,
    }
}

fn append_otlp_http_path(endpoint: &str, path: &str) -> String {
    let endpoint = endpoint.trim_end_matches('/');
    format!("{endpoint}{path}")
}

fn resolve_otlp_endpoint(
    protocol: OtlpProtocol,
    signal_endpoint: Option<String>,
    generic_endpoint: Option<String>,
    http_path: &str,
) -> String {
    if let Some(endpoint) = signal_endpoint.filter(|value| !value.trim().is_empty()) {
        return endpoint;
    }

    match protocol {
        OtlpProtocol::Grpc => generic_endpoint
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_OTLP_ENDPOINT.to_string()),
        OtlpProtocol::HttpProtobuf => append_otlp_http_path(
            generic_endpoint
                .filter(|value| !value.trim().is_empty())
                .as_deref()
                .unwrap_or(DEFAULT_OTLP_HTTP_ENDPOINT),
            http_path,
        ),
    }
}

fn parse_trace_sample_ratio(value: Option<&str>) -> Option<f64> {
    let raw = value?;
    match raw.parse::<f64>() {
        Ok(value) if value.is_finite() && (0.0..=1.0).contains(&value) => Some(value),
        _ => {
            eprintln!(
                "WARNING: invalid OTEL_TRACES_SAMPLE_RATIO '{}'; expected a number between 0.0 and 1.0, keeping default sampler",
                raw
            );
            None
        }
    }
}

fn trace_sample_ratio_from_env() -> Option<f64> {
    parse_trace_sample_ratio(
        std::env::var(env_logging::otlp::OTEL_TRACES_SAMPLE_RATIO)
            .ok()
            .as_deref(),
    )
}

fn build_span_exporter(
    protocol: OtlpProtocol,
    endpoint: &str,
) -> Result<opentelemetry_otlp::SpanExporter, opentelemetry_otlp::ExporterBuildError> {
    match protocol {
        OtlpProtocol::Grpc => opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build(),
        OtlpProtocol::HttpProtobuf => opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .build(),
    }
}

fn build_log_exporter(
    protocol: OtlpProtocol,
    endpoint: &str,
) -> Result<opentelemetry_otlp::LogExporter, opentelemetry_otlp::ExporterBuildError> {
    match protocol {
        OtlpProtocol::Grpc => opentelemetry_otlp::LogExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build(),
        OtlpProtocol::HttpProtobuf => opentelemetry_otlp::LogExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .build(),
    }
}

fn span_events_for_logging() -> FmtSpan {
    if span_events_enabled() {
        FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    }
}

fn log_otel_init_status(service_name: &str, endpoint_opt: Option<(OtlpProtocol, String)>) {
    if let Some((protocol, endpoint)) = endpoint_opt {
        tracing::info!(
            endpoint = %endpoint,
            protocol = %protocol.as_str(),
            service = %service_name,
            "OpenTelemetry OTLP export enabled (traces and logs)"
        );
    } else {
        tracing::info!(
            service = %service_name,
            "OpenTelemetry OTLP export disabled, traces local only"
        );
    }
}

/// Validate a given trace ID according to W3C Trace Context specifications.
/// A valid trace ID is a 32-character hexadecimal string (lowercase).
pub fn is_valid_trace_id(trace_id: &str) -> bool {
    trace_id.len() == 32 && trace_id.chars().all(|c| c.is_ascii_hexdigit())
}

/// Validate a given span ID according to W3C Trace Context specifications.
/// A valid span ID is a 16-character hexadecimal string (lowercase).
pub fn is_valid_span_id(span_id: &str) -> bool {
    span_id.len() == 16 && span_id.chars().all(|c| c.is_ascii_hexdigit())
}

pub struct DistributedTraceIdLayer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTraceContext {
    pub trace_id: String,
    pub span_id: String,
    #[serde(
        default = "default_trace_flags",
        skip_serializing_if = "is_default_trace_flags"
    )]
    pub trace_flags: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracestate: Option<String>,
    #[serde(skip)]
    start: Option<Instant>,
    #[serde(skip)]
    end: Option<Instant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Pending context data collected in on_new_span, to be finalized in on_enter
#[derive(Debug, Clone)]
struct PendingDistributedTraceContext {
    trace_id: Option<String>,
    span_id: Option<String>,
    parent_id: Option<String>,
    trace_flags: Option<String>,
    tracestate: Option<String>,
    x_request_id: Option<String>,
    request_id: Option<String>,
}

/// Macro to emit a tracing event at a dynamic level with a custom target.
macro_rules! emit_at_level {
    ($level:expr, target: $target:expr, $($arg:tt)*) => {
        // tracing::event! requires a compile-time constant level, so we must match
        // on the runtime level and use a literal Level constant in each arm.
        // See: https://github.com/tokio-rs/tracing/issues/2730
        match $level {
            &tracing::Level::ERROR => tracing::event!(target: $target, tracing::Level::ERROR, $($arg)*),
            &tracing::Level::WARN => tracing::event!(target: $target, tracing::Level::WARN, $($arg)*),
            &tracing::Level::INFO => tracing::event!(target: $target, tracing::Level::INFO, $($arg)*),
            &tracing::Level::DEBUG => tracing::event!(target: $target, tracing::Level::DEBUG, $($arg)*),
            &tracing::Level::TRACE => tracing::event!(target: $target, tracing::Level::TRACE, $($arg)*),
        }
    };
}

impl DistributedTraceContext {
    /// Create a traceparent string from the context
    pub fn create_traceparent(&self) -> String {
        format!(
            "00-{}-{}-{}",
            self.trace_id,
            self.span_id,
            normalize_trace_flags(&self.trace_flags)
        )
    }
}

fn default_trace_flags() -> String {
    "01".to_string()
}

fn is_default_trace_flags(trace_flags: &str) -> bool {
    trace_flags == "01"
}

fn is_valid_trace_flags(trace_flags: &str) -> bool {
    trace_flags.len() == 2 && trace_flags.chars().all(|c| c.is_ascii_hexdigit())
}

/// Validate the traceparent version field according to W3C Trace Context.
/// A valid version is a 2-character hex string other than `ff` (forbidden);
/// `00`-`fe` parse, matching the OTel propagator and preserving forward-compat.
fn is_valid_version(version: &str) -> bool {
    version.len() == 2 && matches!(u8::from_str_radix(version, 16), Ok(v) if v != 0xff)
}

fn normalize_trace_flags(trace_flags: &str) -> String {
    if is_valid_trace_flags(trace_flags) {
        trace_flags.to_ascii_lowercase()
    } else {
        default_trace_flags()
    }
}

fn current_otel_trace_flags() -> Option<String> {
    let context = Span::current().context();
    let span = context.span();
    let span_context = span.span_context();
    if !span_context.is_valid() {
        return None;
    }

    Some(
        if span_context.trace_flags().is_sampled() {
            "01"
        } else {
            "00"
        }
        .to_string(),
    )
}

/// Parse a traceparent string into its components
pub fn parse_traceparent(traceparent: &str) -> (Option<String>, Option<String>, Option<String>) {
    let pieces: Vec<_> = traceparent.split('-').collect();
    if pieces.len() != 4 {
        return (None, None, None);
    }
    let version = pieces[0];
    let trace_id = pieces[1];
    let parent_id = pieces[2];
    let trace_flags = pieces[3];

    if !is_valid_version(version)
        || !is_valid_trace_id(trace_id)
        || !is_valid_span_id(parent_id)
        || !is_valid_trace_flags(trace_flags)
    {
        return (None, None, None);
    }

    (
        Some(trace_id.to_string()),
        Some(parent_id.to_string()),
        Some(trace_flags.to_ascii_lowercase()),
    )
}

#[derive(Debug, Clone, Default)]
pub struct TraceParent {
    pub trace_id: Option<String>,
    pub parent_id: Option<String>,
    pub trace_flags: Option<String>,
    pub tracestate: Option<String>,
    pub x_request_id: Option<String>,
    pub request_id: Option<String>,
}

pub trait GenericHeaders {
    fn get(&self, key: &str) -> Option<&str>;
}

impl GenericHeaders for async_nats::HeaderMap {
    fn get(&self, key: &str) -> Option<&str> {
        async_nats::HeaderMap::get(self, key).map(|value| value.as_str())
    }
}

impl GenericHeaders for http::HeaderMap {
    fn get(&self, key: &str) -> Option<&str> {
        http::HeaderMap::get(self, key).and_then(|value| value.to_str().ok())
    }
}

impl TraceParent {
    pub fn from_headers<H: GenericHeaders>(headers: &H) -> TraceParent {
        let mut trace_id = None;
        let mut parent_id = None;
        let mut trace_flags = None;
        let mut tracestate = None;
        let mut x_request_id = None;
        let mut request_id = None;

        if let Some(header_value) = headers.get("traceparent") {
            (trace_id, parent_id, trace_flags) = parse_traceparent(header_value);
        }

        if let Some(header_value) = headers.get("x-request-id") {
            x_request_id = Some(header_value.to_string());
        }

        if let Some(header_value) = headers.get("tracestate") {
            tracestate = Some(header_value.to_string());
        }

        // Read request-id from internal headers, with fallback to deprecated x-dynamo-request-id
        if let Some(header_value) = headers.get("request-id") {
            request_id = Some(header_value.to_string());
        } else if let Some(header_value) = headers.get("x-dynamo-request-id") {
            request_id = Some(header_value.to_string());
        }

        let request_id = request_id.filter(|id| uuid::Uuid::parse_str(id).is_ok());
        TraceParent {
            trace_id,
            parent_id,
            trace_flags,
            tracestate,
            x_request_id,
            request_id,
        }
    }
}

/// Create a span for inference request endpoints (completions, chat, embeddings, etc.).
///
/// Uses `target: "request_span"` which is always allowed through the DYN_LOG filter
/// (via `request_span=trace` directive in `filters()`). This ensures request context
/// (request_id, model, trace_id) is always available on log events.
pub fn make_inference_request_span<B>(req: &Request<B>) -> Span {
    let method = req.method();
    let uri = req.uri();
    let version = format!("{:?}", req.version());
    let trace_parent = TraceParent::from_headers(req.headers());

    let otel_context = extract_otel_context_from_http_headers(req.headers());

    // Ensure every inference request has a request_id on the span.
    // This is the single source of truth — workers and get_or_create_request_id
    // read it back via DistributedTraceIdLayer.
    let request_id = trace_parent
        .request_id
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let span = tracing::info_span!(
            target: "request_span",
        "http-request",
        method = %method,
        uri = %uri,
        version = %version,
        trace_id = trace_parent.trace_id,
        parent_id = trace_parent.parent_id,
        trace_flags = trace_parent.trace_flags,
        x_request_id = trace_parent.x_request_id,
        request_id = %request_id,
        model = tracing::field::Empty,
        input_tokens = tracing::field::Empty,
        output_tokens = tracing::field::Empty,
        image_count = tracing::field::Empty,
        video_count = tracing::field::Empty,
        audio_count = tracing::field::Empty,
        ttft_ms = tracing::field::Empty,
        avg_itl_ms = tracing::field::Empty,
        prefill_worker_id = tracing::field::Empty,
        decode_worker_id = tracing::field::Empty,
    );

    if let Some(context) = otel_context {
        let _ = span.set_parent(context);
    }

    span
}

/// Create a span for system endpoints (health, metrics, models, engine, loras, etc.).
///
/// Same structure as `make_inference_request_span` but uses `target: "system_span"`
/// which follows normal DYN_LOG filtering (debug level by default). The inference
/// span target `request_span` is always-on via a `request_span=trace` directive;
/// system spans are not, keeping high-frequency polling endpoints quiet.
pub fn make_system_request_span<B>(req: &Request<B>) -> Span {
    let method = req.method();
    let uri = req.uri();
    let version = format!("{:?}", req.version());
    let trace_parent = TraceParent::from_headers(req.headers());
    let otel_context = extract_otel_context_from_http_headers(req.headers());

    // Ensure every system request has a request_id on the span.
    let request_id = trace_parent
        .request_id
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let span = tracing::debug_span!(
        target: "system_span",
        "http-request",
        method = %method,
        uri = %uri,
        version = %version,
        trace_id = trace_parent.trace_id,
        parent_id = trace_parent.parent_id,
        trace_flags = trace_parent.trace_flags,
        x_request_id = trace_parent.x_request_id,
        request_id = %request_id,
        model = tracing::field::Empty,
        input_tokens = tracing::field::Empty,
        output_tokens = tracing::field::Empty,
        image_count = tracing::field::Empty,
        video_count = tracing::field::Empty,
        audio_count = tracing::field::Empty,
        ttft_ms = tracing::field::Empty,
        avg_itl_ms = tracing::field::Empty,
        prefill_worker_id = tracing::field::Empty,
        decode_worker_id = tracing::field::Empty,
    );

    if let Some(context) = otel_context {
        let _ = span.set_parent(context);
    }

    span
}

/// Extract OpenTelemetry context from HTTP headers for distributed tracing
fn extract_otel_context_from_http_headers(
    headers: &http::HeaderMap,
) -> Option<opentelemetry::Context> {
    let traceparent_value = headers.get("traceparent")?.to_str().ok()?;

    struct HttpHeaderExtractor<'a>(&'a http::HeaderMap);

    impl<'a> Extractor for HttpHeaderExtractor<'a> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).and_then(|v| v.to_str().ok())
        }

        fn keys(&self) -> Vec<&str> {
            vec!["traceparent", "tracestate"]
                .into_iter()
                .filter(|&key| self.0.get(key).is_some())
                .collect()
        }
    }

    // Early return if traceparent is empty
    if traceparent_value.is_empty() {
        return None;
    }

    let extractor = HttpHeaderExtractor(headers);
    let otel_context = TRACE_PROPAGATOR.extract(&extractor);

    if otel_context.span().span_context().is_valid() {
        Some(otel_context)
    } else {
        None
    }
}

/// Create a handle_payload span from NATS headers with component context
pub fn make_handle_payload_span(
    headers: &async_nats::HeaderMap,
    component: &str,
    endpoint: &str,
    namespace: &str,
    instance_id: u64,
) -> Span {
    let (otel_context, trace_id, parent_span_id) = extract_otel_context_from_nats_headers(headers);
    let trace_parent = TraceParent::from_headers(headers);

    if let (Some(trace_id), Some(parent_id)) = (trace_id.as_ref(), parent_span_id.as_ref()) {
        let span = tracing::info_span!(
            target: "request_span",
            "handle_payload",
            trace_id = trace_id.as_str(),
            parent_id = parent_id.as_str(),
            trace_flags = trace_parent.trace_flags,
            x_request_id = trace_parent.x_request_id,
            request_id = trace_parent.request_id,
            tracestate = trace_parent.tracestate,
            component = component,
            endpoint = endpoint,
            namespace = namespace,
            instance_id = instance_id,
        );

        if let Some(context) = otel_context {
            let _ = span.set_parent(context);
        }
        span
    } else {
        tracing::info_span!(
            target: "request_span",
            "handle_payload",
            trace_flags = trace_parent.trace_flags,
            x_request_id = trace_parent.x_request_id,
            request_id = trace_parent.request_id,
            tracestate = trace_parent.tracestate,
            component = component,
            endpoint = endpoint,
            namespace = namespace,
            instance_id = instance_id,
        )
    }
}

/// Create a handle_payload span from TCP/HashMap headers with component context
pub fn make_handle_payload_span_from_tcp_headers(
    headers: &std::collections::HashMap<String, String>,
    component: &str,
    endpoint: &str,
    namespace: &str,
    instance_id: u64,
) -> Span {
    let (otel_context, trace_id, parent_span_id) = extract_otel_context_from_tcp_headers(headers);
    let x_request_id = headers.get("x-request-id").cloned();
    let request_id = headers
        .get("request-id")
        .or_else(|| headers.get("x-dynamo-request-id"))
        .filter(|id| uuid::Uuid::parse_str(id).is_ok())
        .cloned();
    let tracestate = headers.get("tracestate").cloned();
    let trace_flags = headers.get("traceparent").and_then(|value| {
        let (_, _, flags) = parse_traceparent(value);
        flags
    });

    if let (Some(trace_id), Some(parent_id)) = (trace_id.as_ref(), parent_span_id.as_ref()) {
        let span = tracing::info_span!(
            target: "request_span",
            "handle_payload",
            trace_id = trace_id.as_str(),
            parent_id = parent_id.as_str(),
            trace_flags = trace_flags,
            x_request_id = x_request_id,
            request_id = request_id,
            tracestate = tracestate,
            component = component,
            endpoint = endpoint,
            namespace = namespace,
            instance_id = instance_id,
        );

        if let Some(context) = otel_context {
            let _ = span.set_parent(context);
        }
        span
    } else {
        tracing::info_span!(
            target: "request_span",
            "handle_payload",
            trace_flags = trace_flags,
            x_request_id = x_request_id,
            request_id = request_id,
            tracestate = tracestate,
            component = component,
            endpoint = endpoint,
            namespace = namespace,
            instance_id = instance_id,
        )
    }
}

/// Extract OpenTelemetry trace context from TCP/HashMap headers for distributed tracing
fn extract_otel_context_from_tcp_headers(
    headers: &std::collections::HashMap<String, String>,
) -> (
    Option<opentelemetry::Context>,
    Option<String>,
    Option<String>,
) {
    let traceparent_value = match headers.get("traceparent") {
        Some(value) => value.as_str(),
        None => return (None, None, None),
    };

    let (trace_id, parent_span_id, _) = parse_traceparent(traceparent_value);

    struct TcpHeaderExtractor<'a>(&'a std::collections::HashMap<String, String>);

    impl<'a> Extractor for TcpHeaderExtractor<'a> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).map(|s| s.as_str())
        }

        fn keys(&self) -> Vec<&str> {
            vec!["traceparent", "tracestate"]
                .into_iter()
                .filter(|&key| self.0.get(key).is_some())
                .collect()
        }
    }

    let extractor = TcpHeaderExtractor(headers);
    let otel_context = TRACE_PROPAGATOR.extract(&extractor);

    let context_with_trace = if otel_context.span().span_context().is_valid() {
        Some(otel_context)
    } else {
        None
    };

    (context_with_trace, trace_id, parent_span_id)
}

/// Extract OpenTelemetry trace context from NATS headers for distributed tracing
pub fn extract_otel_context_from_nats_headers(
    headers: &async_nats::HeaderMap,
) -> (
    Option<opentelemetry::Context>,
    Option<String>,
    Option<String>,
) {
    let traceparent_value = match headers.get("traceparent") {
        Some(value) => value.as_str(),
        None => return (None, None, None),
    };

    let (trace_id, parent_span_id, _) = parse_traceparent(traceparent_value);

    struct NatsHeaderExtractor<'a>(&'a async_nats::HeaderMap);

    impl<'a> Extractor for NatsHeaderExtractor<'a> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).map(|value| value.as_str())
        }

        fn keys(&self) -> Vec<&str> {
            vec!["traceparent", "tracestate"]
                .into_iter()
                .filter(|&key| self.0.get(key).is_some())
                .collect()
        }
    }

    let extractor = NatsHeaderExtractor(headers);
    let otel_context = TRACE_PROPAGATOR.extract(&extractor);

    let context_with_trace = if otel_context.span().span_context().is_valid() {
        Some(otel_context)
    } else {
        None
    };

    (context_with_trace, trace_id, parent_span_id)
}

/// Inject OpenTelemetry trace context into NATS headers using W3C Trace Context propagation
pub fn inject_otel_context_into_nats_headers(
    headers: &mut async_nats::HeaderMap,
    context: Option<opentelemetry::Context>,
) {
    let otel_context = context.unwrap_or_else(|| Span::current().context());

    struct NatsHeaderInjector<'a>(&'a mut async_nats::HeaderMap);

    impl<'a> Injector for NatsHeaderInjector<'a> {
        fn set(&mut self, key: &str, value: String) {
            self.0.insert(key, value);
        }
    }

    let mut injector = NatsHeaderInjector(headers);
    TRACE_PROPAGATOR.inject_context(&otel_context, &mut injector);
}

/// Inject trace context from current span into NATS headers
pub fn inject_current_trace_into_nats_headers(headers: &mut async_nats::HeaderMap) {
    inject_otel_context_into_nats_headers(headers, None);
}

// Inject trace headers into a generic HashMap for HTTP/TCP transports
pub fn inject_trace_headers_into_map(headers: &mut std::collections::HashMap<String, String>) {
    if let Some(trace_context) = get_distributed_tracing_context() {
        // Inject W3C traceparent header
        headers.insert(
            "traceparent".to_string(),
            trace_context.create_traceparent(),
        );

        // Inject optional tracestate
        if let Some(tracestate) = trace_context.tracestate {
            headers.insert("tracestate".to_string(), tracestate);
        }

        // Inject custom request IDs
        if let Some(x_request_id) = trace_context.x_request_id {
            headers.insert("x-request-id".to_string(), x_request_id);
        }
        if let Some(request_id) = trace_context.request_id {
            headers.insert("request-id".to_string(), request_id);
        }
    }
}

/// Create a client_request span linked to the parent trace context
pub fn make_client_request_span(
    operation: &str,
    request_id: &str,
    trace_context: Option<&DistributedTraceContext>,
    instance_id: Option<&str>,
) -> Span {
    if let Some(ctx) = trace_context {
        let mut headers = async_nats::HeaderMap::new();
        headers.insert("traceparent", ctx.create_traceparent());

        if let Some(ref tracestate) = ctx.tracestate {
            headers.insert("tracestate", tracestate.as_str());
        }

        let (otel_context, _extracted_trace_id, _extracted_parent_span_id) =
            extract_otel_context_from_nats_headers(&headers);

        let span = if let Some(inst_id) = instance_id {
            tracing::info_span!(
                "client_request",
                operation = operation,
                request_id = request_id,
                instance_id = inst_id,
                trace_id = ctx.trace_id.as_str(),
                parent_id = ctx.span_id.as_str(),
                trace_flags = ctx.trace_flags.as_str(),
                x_request_id = ctx.x_request_id.as_deref(),
            )
        } else {
            tracing::info_span!(
                "client_request",
                operation = operation,
                request_id = request_id,
                trace_id = ctx.trace_id.as_str(),
                parent_id = ctx.span_id.as_str(),
                trace_flags = ctx.trace_flags.as_str(),
                x_request_id = ctx.x_request_id.as_deref(),
            )
        };

        if let Some(context) = otel_context {
            let _ = span.set_parent(context);
        }

        span
    } else if let Some(inst_id) = instance_id {
        tracing::info_span!(
            "client_request",
            operation = operation,
            request_id = request_id,
            instance_id = inst_id,
        )
    } else {
        tracing::info_span!(
            "client_request",
            operation = operation,
            request_id = request_id,
        )
    }
}

#[derive(Debug, Default)]
pub struct FieldVisitor {
    pub fields: HashMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields
            .insert(field.name().to_string(), format!("{:?}", value).to_string());
    }
}

impl<S> Layer<S> for DistributedTraceIdLayer
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    // Capture close span time
    // Currently not used but added for future use in timing
    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            let mut extensions = span.extensions_mut();
            if let Some(distributed_tracing_context) =
                extensions.get_mut::<DistributedTraceContext>()
            {
                distributed_tracing_context.end = Some(Instant::now());
            }
        }
    }

    // Collects span attributes and metadata in on_new_span
    // Final initialization deferred to on_enter when OtelData is available
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut trace_id: Option<String> = None;
            let mut parent_id: Option<String> = None;
            let mut span_id: Option<String> = None;
            let mut trace_flags: Option<String> = None;
            let mut x_request_id: Option<String> = None;
            let mut request_id: Option<String> = None;
            let mut tracestate: Option<String> = None;
            let mut visitor = FieldVisitor::default();
            attrs.record(&mut visitor);

            // Extract trace_id from span attributes
            if let Some(trace_id_input) = visitor.fields.get("trace_id") {
                if !is_valid_trace_id(trace_id_input) {
                    tracing::trace!("trace id  '{trace_id_input}' is not valid! Ignoring.");
                } else {
                    trace_id = Some(trace_id_input.to_string());
                }
            }

            // Extract span_id from span attributes
            if let Some(span_id_input) = visitor.fields.get("span_id") {
                if !is_valid_span_id(span_id_input) {
                    tracing::trace!("span id  '{span_id_input}' is not valid! Ignoring.");
                } else {
                    span_id = Some(span_id_input.to_string());
                }
            }

            // Extract parent_id from span attributes
            if let Some(parent_id_input) = visitor.fields.get("parent_id") {
                if !is_valid_span_id(parent_id_input) {
                    tracing::trace!("parent id  '{parent_id_input}' is not valid! Ignoring.");
                } else {
                    parent_id = Some(parent_id_input.to_string());
                }
            }

            if let Some(trace_flags_input) = visitor.fields.get("trace_flags") {
                if !is_valid_trace_flags(trace_flags_input) {
                    tracing::trace!("trace flags '{trace_flags_input}' are not valid! Ignoring.");
                } else {
                    trace_flags = Some(trace_flags_input.to_ascii_lowercase());
                }
            }

            // Extract tracestate
            if let Some(tracestate_input) = visitor.fields.get("tracestate") {
                tracestate = Some(tracestate_input.to_string());
            }

            // Extract x_request_id
            if let Some(x_request_id_input) = visitor.fields.get("x_request_id") {
                x_request_id = Some(x_request_id_input.to_string());
            }

            // Extract request_id (with backward compat for x_dynamo_request_id)
            if let Some(request_id_input) = visitor.fields.get("request_id") {
                request_id = Some(request_id_input.to_string());
            } else if let Some(x_request_id_input) = visitor.fields.get("x_dynamo_request_id") {
                request_id = Some(x_request_id_input.to_string());
            }

            // Inherit trace context from parent span if available
            if parent_id.is_none()
                && let Some(parent_span_id) = ctx.current_span().id()
                && let Some(parent_span) = ctx.span(parent_span_id)
            {
                let parent_ext = parent_span.extensions();
                if let Some(parent_tracing_context) = parent_ext.get::<DistributedTraceContext>() {
                    trace_id = Some(parent_tracing_context.trace_id.clone());
                    parent_id = Some(parent_tracing_context.span_id.clone());
                    if trace_flags.is_none() {
                        trace_flags = Some(parent_tracing_context.trace_flags.clone());
                    }
                    tracestate = parent_tracing_context.tracestate.clone();
                    if x_request_id.is_none() {
                        x_request_id = parent_tracing_context.x_request_id.clone();
                    }
                    if request_id.is_none() {
                        request_id = parent_tracing_context.request_id.clone();
                    }
                }
            }

            // Validate consistency
            if (parent_id.is_some() || span_id.is_some()) && trace_id.is_none() {
                tracing::error!("parent id or span id are set but trace id is not set!");
                // Clear inconsistent IDs to maintain trace integrity
                parent_id = None;
                span_id = None;
            }

            // Store pending context - will be finalized in on_enter
            let mut extensions = span.extensions_mut();
            extensions.insert(PendingDistributedTraceContext {
                trace_id,
                span_id,
                parent_id,
                trace_flags,
                tracestate,
                x_request_id,
                request_id,
            });
        }
    }

    // Finalizes the DistributedTraceContext when span is entered
    // At this point, OtelData should have valid trace_id and span_id
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            // Check if already initialized (e.g., span re-entered)
            {
                let extensions = span.extensions();
                if extensions.get::<DistributedTraceContext>().is_some() {
                    return;
                }
            }

            // Get the pending context and extract OtelData IDs
            let mut extensions = span.extensions_mut();
            let pending = match extensions.remove::<PendingDistributedTraceContext>() {
                Some(p) => p,
                None => {
                    // This shouldn't happen - on_new_span should have created it
                    tracing::error!("PendingDistributedTraceContext not found in on_enter");
                    return;
                }
            };

            let mut trace_id = pending.trace_id;
            let mut span_id = pending.span_id;
            let parent_id = pending.parent_id;
            let mut trace_flags = pending.trace_flags;
            let tracestate = pending.tracestate;
            let x_request_id = pending.x_request_id;
            let request_id = pending.request_id;

            // Try to extract from OtelData if not already set
            // Need to drop extensions_mut to get immutable borrow for OtelData
            drop(extensions);

            if trace_id.is_none() || span_id.is_none() {
                let extensions = span.extensions();
                if let Some(otel_data) = extensions.get::<tracing_opentelemetry::OtelData>() {
                    // Extract trace_id from OTEL data if not already set
                    if trace_id.is_none()
                        && let Some(otel_trace_id) = otel_data.trace_id()
                    {
                        let trace_id_str = format!("{}", otel_trace_id);
                        if is_valid_trace_id(&trace_id_str) {
                            trace_id = Some(trace_id_str);
                        }
                    }

                    // Extract span_id from OTEL data if not already set
                    if span_id.is_none()
                        && let Some(otel_span_id) = otel_data.span_id()
                    {
                        let span_id_str = format!("{}", otel_span_id);
                        if is_valid_span_id(&span_id_str) {
                            span_id = Some(span_id_str);
                        }
                    }
                }
            }

            if trace_flags.is_none() {
                trace_flags = current_otel_trace_flags();
            }

            // Panic if we still don't have required IDs
            if trace_id.is_none() {
                panic!(
                    "trace_id is not set in on_enter - OtelData may not be properly initialized"
                );
            }

            if span_id.is_none() {
                panic!("span_id is not set in on_enter - OtelData may not be properly initialized");
            }

            let span_level = span.metadata().level();
            let mut extensions = span.extensions_mut();
            extensions.insert(DistributedTraceContext {
                trace_id: trace_id.expect("Trace ID must be set"),
                span_id: span_id.expect("Span ID must be set"),
                trace_flags: trace_flags.unwrap_or_else(default_trace_flags),
                parent_id,
                tracestate,
                start: Some(Instant::now()),
                end: None,
                x_request_id,
                request_id,
            });

            drop(extensions);

            // Emit SPAN_FIRST_ENTRY event. This only runs if the span passed the layer's filter
            // (on_enter is not called for filtered-out spans), so no additional check needed.
            if span_events_enabled() {
                emit_at_level!(span_level, target: "span_event", message = "SPAN_FIRST_ENTRY");
            }
        }
    }
}

// Enables functions to retreive their current
// context for adding to distributed headers
pub fn get_distributed_tracing_context() -> Option<DistributedTraceContext> {
    Span::current()
        .with_subscriber(|(id, subscriber)| {
            subscriber
                .downcast_ref::<Registry>()
                .and_then(|registry| registry.span_data(id))
                .and_then(|span_data| {
                    let extensions = span_data.extensions();
                    extensions.get::<DistributedTraceContext>().cloned()
                })
        })
        .flatten()
        .map(|mut context| {
            // Propagate this node's live OTel sampling decision (W3C: `sampled`
            // reflects the immediate caller, not the original client), so a
            // non-parent sampler overrides the inbound flag downstream.
            if let Some(trace_flags) = current_otel_trace_flags() {
                context.trace_flags = trace_flags;
            }
            context
        })
}

/// Initialize the logger - must be called when Tokio runtime is available
pub fn init() {
    INIT.call_once(|| {
        if let Err(e) = setup_logging() {
            eprintln!("Failed to initialize logging: {}", e);
            std::process::exit(1);
        }
    });
}

#[cfg(feature = "tokio-console")]
fn setup_logging() -> Result<(), Box<dyn std::error::Error>> {
    let tokio_console_layer = console_subscriber::ConsoleLayer::builder()
        .with_default_env()
        .server_addr(([0, 0, 0, 0], console_subscriber::Server::DEFAULT_PORT))
        .spawn();
    let tokio_console_target = tracing_subscriber::filter::Targets::new()
        .with_default(LevelFilter::ERROR)
        .with_target("runtime", LevelFilter::TRACE)
        .with_target("tokio", LevelFilter::TRACE);
    let l = fmt::layer()
        .with_ansi(!disable_ansi_logging())
        .event_format(fmt::format().compact().with_timer(TimeFormatter::new()))
        .with_writer(std::io::stderr)
        .with_filter(filters(load_config()));
    tracing_subscriber::registry()
        .with(l)
        .with(tokio_console_layer.with_filter(tokio_console_target))
        .init();
    Ok(())
}

#[cfg(not(feature = "tokio-console"))]
fn setup_logging() -> Result<(), Box<dyn std::error::Error>> {
    let fmt_filter_layer = filters(load_config());
    let trace_filter_layer = filters(load_config());
    let otel_filter_layer = filters(load_config());
    let otel_logs_filter_layer = filters(load_config());
    let jsonl_enabled = jsonl_logging_enabled();
    let otlp_enabled = otlp_exporter_enabled();

    if jsonl_enabled || otlp_enabled {
        let service_name = get_service_name();
        let sample_ratio = trace_sample_ratio_from_env();

        // Build tracer and logger providers - with or without OTLP export
        let (tracer_provider, logger_provider_opt, endpoint_opt) = if otlp_enabled {
            // Export enabled: create OTLP exporters with batch processors
            let protocol = otlp_protocol_from_env();
            let traces_protocol = resolve_signal_otlp_protocol(
                protocol,
                std::env::var(env_logging::otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL)
                    .ok()
                    .as_deref(),
                env_logging::otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
            );
            let logs_protocol = resolve_signal_otlp_protocol(
                protocol,
                std::env::var(env_logging::otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL)
                    .ok()
                    .as_deref(),
                env_logging::otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL,
            );
            let generic_endpoint =
                std::env::var(env_logging::otlp::OTEL_EXPORTER_OTLP_ENDPOINT).ok();
            let traces_endpoint_env =
                std::env::var(env_logging::otlp::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT).ok();
            let logs_endpoint_env =
                std::env::var(env_logging::otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT).ok();
            let traces_endpoint = resolve_otlp_endpoint(
                traces_protocol,
                traces_endpoint_env,
                generic_endpoint.clone(),
                "/v1/traces",
            );
            let logs_endpoint = resolve_otlp_endpoint(
                logs_protocol,
                logs_endpoint_env,
                generic_endpoint,
                "/v1/logs",
            );

            let resource = opentelemetry_sdk::Resource::builder_empty()
                .with_service_name(service_name.clone())
                .build();

            let span_exporter = build_span_exporter(traces_protocol, &traces_endpoint)?;

            let mut tracer_provider_builder =
                opentelemetry_sdk::trace::SdkTracerProvider::builder()
                    .with_batch_exporter(span_exporter)
                    .with_resource(resource.clone());
            if let Some(sample_ratio) = sample_ratio {
                tracer_provider_builder = tracer_provider_builder.with_sampler(
                    Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(sample_ratio))),
                );
            }
            let tracer_provider = tracer_provider_builder.build();

            let log_exporter = build_log_exporter(logs_protocol, &logs_endpoint)?;

            let logger_provider = SdkLoggerProvider::builder()
                .with_batch_exporter(log_exporter)
                .with_resource(resource)
                .build();

            (
                tracer_provider,
                Some(logger_provider),
                Some((traces_protocol, traces_endpoint)),
            )
        } else {
            // No export - traces generated locally only (for logging/trace IDs)
            let mut provider_builder = opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_resource(
                    opentelemetry_sdk::Resource::builder_empty()
                        .with_service_name(service_name.clone())
                        .build(),
                );
            if let Some(sample_ratio) = sample_ratio {
                provider_builder = provider_builder.with_sampler(Sampler::ParentBased(Box::new(
                    Sampler::TraceIdRatioBased(sample_ratio),
                )));
            }
            let provider = provider_builder.build();

            (provider, None, None)
        };

        // Register the provider globally so direct OTel API users
        // (`opentelemetry::global::tracer(...)`) hit the same exporter as
        // the tracing-opentelemetry bridge below. Without this, ad-hoc
        // OTel spans created via `global::tracer()` go to the default
        // no-op provider and are silently dropped.
        // Cheap — `SdkTracerProvider` is Arc-shared internally.
        opentelemetry::global::set_tracer_provider(tracer_provider.clone());

        let tracer = tracer_provider.tracer(service_name.to_string());
        let otel_logs_layer = logger_provider_opt
            .as_ref()
            .map(|lp| OpenTelemetryTracingBridge::new(lp).with_filter(otel_logs_filter_layer));

        macro_rules! init_otel_subscriber {
            ($fmt_layer:expr) => {
                tracing_subscriber::registry()
                    .with(
                        tracing_opentelemetry::layer()
                            .with_tracer(tracer)
                            .with_filter(otel_filter_layer),
                    )
                    .with(otel_logs_layer)
                    .with(DistributedTraceIdLayer.with_filter(trace_filter_layer))
                    .with($fmt_layer)
                    .init();
            };
        }

        if jsonl_enabled {
            let l = fmt::layer()
                .with_ansi(false)
                .with_span_events(span_events_for_logging())
                .event_format(CustomJsonFormatter::new())
                .with_writer(std::io::stderr)
                .with_filter(fmt_filter_layer);
            init_otel_subscriber!(l);
        } else {
            let l = fmt::layer()
                .with_ansi(!disable_ansi_logging())
                .event_format(fmt::format().compact().with_timer(TimeFormatter::new()))
                .with_writer(std::io::stderr)
                .with_filter(fmt_filter_layer);
            init_otel_subscriber!(l);
        }

        log_otel_init_status(&service_name, endpoint_opt);
    } else {
        let l = fmt::layer()
            .with_ansi(!disable_ansi_logging())
            .event_format(fmt::format().compact().with_timer(TimeFormatter::new()))
            .with_writer(std::io::stderr)
            .with_filter(fmt_filter_layer);

        tracing_subscriber::registry().with(l).init();
    }

    Ok(())
}

fn filters(config: LoggingConfig) -> EnvFilter {
    let mut filter_layer = EnvFilter::builder()
        .with_default_directive(config.log_level.parse().unwrap())
        .with_env_var(env_logging::DYN_LOG)
        .from_env_lossy();

    for (module, level) in config.log_filters {
        match format!("{module}={level}").parse::<Directive>() {
            Ok(d) => {
                filter_layer = filter_layer.add_directive(d);
            }
            Err(e) => {
                eprintln!("Failed parsing filter '{level}' for module '{module}': {e}");
            }
        }
    }

    // When span events are enabled, allow "span_event" target at all levels
    // This ensures SPAN_FIRST_ENTRY events pass the filter when emitted from on_enter
    if span_events_enabled() {
        filter_layer = filter_layer.add_directive("span_event=trace".parse().unwrap());
    }

    // Always allow infrastructure request spans regardless of DYN_LOG level.
    // This ensures request context (request_id, model, trace_id) is always
    // available on log events, even when DYN_LOG=error or DYN_LOG=warn.
    // Can be overridden via DYN_LOG=request_span=<level> if needed.
    filter_layer = filter_layer.add_directive("request_span=trace".parse().unwrap());

    filter_layer
}

/// Log a message with file and line info
/// Used by Python wrapper
pub fn log_message(level: &str, message: &str, module: &str, file: &str, line: u32) {
    let level = match level {
        "debug" => log::Level::Debug,
        "info" => log::Level::Info,
        "warn" => log::Level::Warn,
        "error" => log::Level::Error,
        "warning" => log::Level::Warn,
        _ => log::Level::Info,
    };
    log::logger().log(
        &log::Record::builder()
            .args(format_args!("{}", message))
            .level(level)
            .target(module)
            .file(Some(file))
            .line(Some(line))
            .build(),
    );
}

fn load_config() -> LoggingConfig {
    let config_path =
        std::env::var(env_logging::DYN_LOGGING_CONFIG_PATH).unwrap_or_else(|_| "".to_string());
    let figment = Figment::new()
        .merge(Serialized::defaults(LoggingConfig::default()))
        .merge(Toml::file("/opt/dynamo/etc/logging.toml"))
        .merge(Toml::file(config_path));

    figment.extract().unwrap()
}

#[derive(Serialize)]
struct JsonLog<'a> {
    time: String,
    level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<u32>,
    target: String,
    message: serde_json::Value,
    #[serde(flatten)]
    fields: BTreeMap<String, serde_json::Value>,
}

struct TimeFormatter {
    use_local_tz: bool,
}

impl TimeFormatter {
    fn new() -> Self {
        Self {
            use_local_tz: crate::config::use_local_timezone(),
        }
    }

    fn format_now(&self) -> String {
        if self.use_local_tz {
            chrono::Local::now()
                .format("%Y-%m-%dT%H:%M:%S%.6f%:z")
                .to_string()
        } else {
            chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%S%.6fZ")
                .to_string()
        }
    }
}

impl FormatTime for TimeFormatter {
    fn format_time(&self, w: &mut fmt::format::Writer<'_>) -> std::fmt::Result {
        write!(w, "{}", self.format_now())
    }
}

struct CustomJsonFormatter {
    time_formatter: TimeFormatter,
}

impl CustomJsonFormatter {
    fn new() -> Self {
        Self {
            time_formatter: TimeFormatter::new(),
        }
    }
}

use once_cell::sync::Lazy;
use regex::Regex;

/// Static W3C Trace Context propagator instance to avoid repeated allocations
static TRACE_PROPAGATOR: Lazy<opentelemetry_sdk::propagation::TraceContextPropagator> =
    Lazy::new(opentelemetry_sdk::propagation::TraceContextPropagator::new);

fn parse_tracing_duration(s: &str) -> Option<u64> {
    static RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r#"^["']?\s*([0-9.]+)\s*(µs|us|ns|ms|s)\s*["']?$"#).unwrap());
    let captures = RE.captures(s)?;
    let value: f64 = captures[1].parse().ok()?;
    let unit = &captures[2];
    match unit {
        "ns" => Some((value / 1000.0) as u64),
        "µs" | "us" => Some(value as u64),
        "ms" => Some((value * 1000.0) as u64),
        "s" => Some((value * 1_000_000.0) as u64),
        _ => None,
    }
}

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for CustomJsonFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        let mut visitor = JsonVisitor::default();
        let time = self.time_formatter.format_now();
        event.record(&mut visitor);
        let mut message = visitor
            .fields
            .remove("message")
            .unwrap_or(serde_json::Value::String("".to_string()));

        let mut target_override: Option<String> = None;

        let current_span = event
            .parent()
            .and_then(|id| ctx.span(id))
            .or_else(|| ctx.lookup_current());
        if let Some(span) = current_span {
            let ext = span.extensions();
            let data = ext.get::<FormattedFields<N>>().unwrap();
            let span_fields: Vec<(&str, &str)> = data
                .fields
                .split(' ')
                .filter_map(|entry| entry.split_once('='))
                .collect();
            for (name, value) in span_fields {
                visitor.fields.insert(
                    name.to_string(),
                    serde_json::Value::String(value.trim_matches('"').to_string()),
                );
            }

            let busy_us = visitor
                .fields
                .remove("time.busy")
                .and_then(|v| parse_tracing_duration(&v.to_string()));
            let idle_us = visitor
                .fields
                .remove("time.idle")
                .and_then(|v| parse_tracing_duration(&v.to_string()));

            if let (Some(busy_us), Some(idle_us)) = (busy_us, idle_us) {
                visitor.fields.insert(
                    "time.busy_us".to_string(),
                    serde_json::Value::Number(busy_us.into()),
                );
                visitor.fields.insert(
                    "time.idle_us".to_string(),
                    serde_json::Value::Number(idle_us.into()),
                );
                visitor.fields.insert(
                    "time.duration_us".to_string(),
                    serde_json::Value::Number((busy_us + idle_us).into()),
                );
            }

            let is_span_created = message.as_str() == Some("SPAN_FIRST_ENTRY");
            let is_span_closed = message.as_str() == Some("close");
            if is_span_created || is_span_closed {
                target_override = Some(span.metadata().target().to_string());
                if is_span_closed {
                    message = serde_json::Value::String("SPAN_CLOSED".to_string());
                }
            }

            visitor.fields.insert(
                "span_name".to_string(),
                serde_json::Value::String(span.name().to_string()),
            );

            if let Some(tracing_context) = ext.get::<DistributedTraceContext>() {
                visitor.fields.insert(
                    "span_id".to_string(),
                    serde_json::Value::String(tracing_context.span_id.clone()),
                );
                visitor.fields.insert(
                    "trace_id".to_string(),
                    serde_json::Value::String(tracing_context.trace_id.clone()),
                );
                if let Some(parent_id) = tracing_context.parent_id.clone() {
                    visitor.fields.insert(
                        "parent_id".to_string(),
                        serde_json::Value::String(parent_id),
                    );
                } else {
                    visitor.fields.remove("parent_id");
                }
                if let Some(tracestate) = tracing_context.tracestate.clone() {
                    visitor.fields.insert(
                        "tracestate".to_string(),
                        serde_json::Value::String(tracestate),
                    );
                } else {
                    visitor.fields.remove("tracestate");
                }
                if let Some(x_request_id) = tracing_context.x_request_id.clone() {
                    visitor.fields.insert(
                        "x_request_id".to_string(),
                        serde_json::Value::String(x_request_id),
                    );
                } else {
                    visitor.fields.remove("x_request_id");
                }

                if let Some(request_id) = tracing_context.request_id.clone() {
                    visitor.fields.insert(
                        "request_id".to_string(),
                        serde_json::Value::String(request_id),
                    );
                } else {
                    visitor.fields.remove("request_id");
                }
                // Remove old field name if present
                visitor.fields.remove("x_dynamo_request_id");
            } else {
                tracing::error!(
                    "Distributed Trace Context not found, falling back to internal ids"
                );
                visitor.fields.insert(
                    "span_id".to_string(),
                    serde_json::Value::String(span.id().into_u64().to_string()),
                );
                if let Some(parent) = span.parent() {
                    visitor.fields.insert(
                        "parent_id".to_string(),
                        serde_json::Value::String(parent.id().into_u64().to_string()),
                    );
                }
            }
        } else {
            let reserved_fields = [
                "trace_id",
                "span_id",
                "parent_id",
                "span_name",
                "tracestate",
            ];
            for reserved_field in reserved_fields {
                visitor.fields.remove(reserved_field);
            }
        }
        let metadata = event.metadata();
        let log = JsonLog {
            level: metadata.level().to_string(),
            time,
            file: metadata.file(),
            line: metadata.line(),
            target: target_override.unwrap_or_else(|| metadata.target().to_string()),
            message,
            fields: visitor.fields,
        };
        let json = serde_json::to_string(&log).unwrap();
        writeln!(writer, "{json}")
    }
}

#[derive(Default)]
struct JsonVisitor {
    fields: BTreeMap<String, serde_json::Value>,
}

impl tracing::field::Visit for JsonVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::Value::String(format!("{value:?}")),
        );
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() != "message" {
            match serde_json::from_str::<Value>(value) {
                Ok(json_val) => self.fields.insert(field.name().to_string(), json_val),
                Err(_) => self.fields.insert(field.name().to_string(), value.into()),
            };
        } else {
            self.fields.insert(field.name().to_string(), value.into());
        }
    }

    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.fields
            .insert(field.name().to_string(), serde_json::Value::Bool(value));
    }

    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::Value::Number(value.into()),
        );
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::Value::Number(value.into()),
        );
    }

    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        use serde_json::value::Number;
        self.fields.insert(
            field.name().to_string(),
            serde_json::Value::Number(Number::from_f64(value).unwrap_or(0.into())),
        );
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use anyhow::{Result, anyhow};
    use chrono::{DateTime, Utc};
    use jsonschema::{Draft, JSONSchema};
    use serde_json::Value;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use stdio_override::*;
    use tempfile::NamedTempFile;

    #[test]
    fn otlp_protocol_defaults_to_grpc() {
        assert_eq!(parse_otlp_protocol(None), OtlpProtocol::Grpc);
        assert_eq!(parse_otlp_protocol(Some("")), OtlpProtocol::Grpc);
        assert_eq!(parse_otlp_protocol(Some("grpc")), OtlpProtocol::Grpc);
        assert_eq!(
            parse_otlp_protocol(Some("http/protobuf")),
            OtlpProtocol::HttpProtobuf
        );
        assert_eq!(
            parse_otlp_protocol(Some("HTTP/PROTOBUF")),
            OtlpProtocol::HttpProtobuf
        );
        assert_eq!(parse_otlp_protocol(Some("bad")), OtlpProtocol::Grpc);
    }

    #[test]
    fn otlp_signal_protocol_overrides_generic_protocol() {
        let generic_protocol = OtlpProtocol::Grpc;
        assert_eq!(
            resolve_signal_otlp_protocol(
                generic_protocol,
                Some("http/protobuf"),
                env_logging::otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
            ),
            OtlpProtocol::HttpProtobuf
        );
        assert_eq!(
            resolve_signal_otlp_protocol(
                generic_protocol,
                Some(""),
                env_logging::otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
            ),
            OtlpProtocol::Grpc
        );
        assert_eq!(
            resolve_signal_otlp_protocol(
                generic_protocol,
                None,
                env_logging::otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
            ),
            OtlpProtocol::Grpc
        );
    }

    #[test]
    fn otlp_http_endpoint_appends_signal_paths_from_generic_endpoint() {
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::HttpProtobuf,
                None,
                Some("https://llm-observe.weizhipin.com".to_string()),
                "/v1/traces",
            ),
            "https://llm-observe.weizhipin.com/v1/traces"
        );
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::HttpProtobuf,
                None,
                Some("https://llm-observe.weizhipin.com/".to_string()),
                "/v1/logs",
            ),
            "https://llm-observe.weizhipin.com/v1/logs"
        );
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::HttpProtobuf,
                None,
                Some("https://llm-observe.weizhipin.com/v1/traces".to_string()),
                "/v1/traces",
            ),
            "https://llm-observe.weizhipin.com/v1/traces/v1/traces"
        );
    }

    #[test]
    fn otlp_signal_endpoint_is_used_verbatim() {
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::HttpProtobuf,
                Some("https://collector.example/custom/traces".to_string()),
                Some("https://collector.example".to_string()),
                "/v1/traces",
            ),
            "https://collector.example/custom/traces"
        );
    }

    #[test]
    fn otlp_grpc_endpoint_keeps_generic_endpoint_verbatim() {
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::Grpc,
                None,
                Some("http://otel-collector:4317".to_string()),
                "/v1/traces",
            ),
            "http://otel-collector:4317"
        );
    }

    #[test]
    fn trace_sample_ratio_is_optional_and_bounded() {
        assert_eq!(parse_trace_sample_ratio(None), None);
        assert_eq!(parse_trace_sample_ratio(Some("0")), Some(0.0));
        assert_eq!(parse_trace_sample_ratio(Some("0.01")), Some(0.01));
        assert_eq!(parse_trace_sample_ratio(Some("1")), Some(1.0));
        assert_eq!(parse_trace_sample_ratio(Some("-0.1")), None);
        assert_eq!(parse_trace_sample_ratio(Some("1.1")), None);
        assert_eq!(parse_trace_sample_ratio(Some("nan")), None);
        assert_eq!(parse_trace_sample_ratio(Some("bad")), None);
    }

    static LOG_LINE_SCHEMA: &str = r#"
    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "title": "Runtime Log Line",
      "type": "object",
      "required": [
        "file",
        "level",
        "line",
        "message",
        "target",
        "time"
      ],
      "properties": {
        "file":      { "type": "string" },
        "level":     { "type": "string", "enum": ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"] },
        "line":      { "type": "integer" },
        "message":   { "type": "string" },
        "target":    { "type": "string" },
        "time":      { "type": "string", "format": "date-time" },
        "span_id":   { "type": "string", "pattern": "^[a-f0-9]{16}$" },
        "parent_id": { "type": "string", "pattern": "^[a-f0-9]{16}$" },
        "trace_id":  { "type": "string", "pattern": "^[a-f0-9]{32}$" },
        "span_name": { "type": "string" },
        "time.busy_us":     { "type": "integer" },
        "time.duration_us": { "type": "integer" },
        "time.idle_us":     { "type": "integer" },
        "tracestate": { "type": "string" }
      },
      "additionalProperties": true
    }
    "#;

    #[tracing::instrument(skip_all)]
    async fn parent() {
        tracing::trace!(message = "parent!");
        if let Some(my_ctx) = get_distributed_tracing_context() {
            tracing::info!(my_trace_id = my_ctx.trace_id);
        }
        child().await;
    }

    #[tracing::instrument(skip_all)]
    async fn child() {
        tracing::trace!(message = "child");
        if let Some(my_ctx) = get_distributed_tracing_context() {
            tracing::info!(my_trace_id = my_ctx.trace_id);
        }
        grandchild().await;
    }

    #[tracing::instrument(skip_all)]
    async fn grandchild() {
        tracing::trace!(message = "grandchild");
        if let Some(my_ctx) = get_distributed_tracing_context() {
            tracing::info!(my_trace_id = my_ctx.trace_id);
        }
    }

    pub fn load_log(file_name: &str) -> Result<Vec<serde_json::Value>> {
        let schema_json: Value =
            serde_json::from_str(LOG_LINE_SCHEMA).expect("schema parse failure");
        let compiled_schema = JSONSchema::options()
            .with_draft(Draft::Draft7)
            .compile(&schema_json)
            .expect("Invalid schema");

        let f = File::open(file_name)?;
        let reader = BufReader::new(f);
        let mut result = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let val: Value = serde_json::from_str(&line)
                .map_err(|e| anyhow!("Line {}: invalid JSON: {}", line_num + 1, e))?;

            if let Err(errors) = compiled_schema.validate(&val) {
                let errs = errors.map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
                return Err(anyhow!(
                    "Line {}: JSON Schema Validation errors: {}",
                    line_num + 1,
                    errs
                ));
            }
            println!("{}", val);
            result.push(val);
        }
        Ok(result)
    }

    // Field validators (W3C Trace Context): each rule is tested directly here.
    // The parse_traceparent tests below only cover parsing/structure + wiring,
    // not the per-field rules.

    #[test]
    fn is_valid_version_accepts_00_to_fe_rejects_ff_and_malformed() {
        assert!(is_valid_version("00"));
        assert!(is_valid_version("01"));
        assert!(is_valid_version("fe")); // highest valid version
        assert!(!is_valid_version("ff")); // forbidden by W3C
        assert!(!is_valid_version("FF")); // uppercase ff is still 0xff
        assert!(!is_valid_version("zz")); // non-hex
        assert!(!is_valid_version("0")); // too short
        assert!(!is_valid_version("000")); // too long
        assert!(!is_valid_version("")); // empty
    }

    #[test]
    fn is_valid_trace_id_requires_32_hex() {
        assert!(is_valid_trace_id(&"a".repeat(32)));
        assert!(is_valid_trace_id("0123456789abcdefABCDEF0123456789")); // case-insensitive
        assert!(!is_valid_trace_id(&"1".repeat(31))); // too short
        assert!(!is_valid_trace_id(&"1".repeat(33))); // too long
        assert!(!is_valid_trace_id(&format!("{}g", "1".repeat(31)))); // non-hex
        assert!(!is_valid_trace_id("")); // empty
    }

    #[test]
    fn is_valid_span_id_requires_16_hex() {
        assert!(is_valid_span_id(&"2".repeat(16)));
        assert!(!is_valid_span_id(&"2".repeat(15))); // too short
        assert!(!is_valid_span_id(&"2".repeat(17))); // too long
        assert!(!is_valid_span_id(&format!("{}g", "2".repeat(15)))); // non-hex
        assert!(!is_valid_span_id("")); // empty
    }

    #[test]
    fn is_valid_trace_flags_requires_2_hex() {
        assert!(is_valid_trace_flags("00"));
        assert!(is_valid_trace_flags("ff")); // any 2 hex digits are structurally valid
        assert!(is_valid_trace_flags("0A")); // case-insensitive
        assert!(!is_valid_trace_flags("0")); // too short
        assert!(!is_valid_trace_flags("000")); // too long
        assert!(!is_valid_trace_flags("0x")); // non-hex
    }

    #[test]
    fn parse_traceparent_happy_path() {
        // Fields extracted by position; trace_flags is lowercased.
        assert_eq!(
            parse_traceparent("00-11111111111111111111111111111111-2222222222222222-0A"),
            (
                Some("11111111111111111111111111111111".to_string()),
                Some("2222222222222222".to_string()),
                Some("0a".to_string()), // lowercased
            )
        );

        // A future, same-shape version (00-fe) still parses (forward-compat).
        let (trace_id, _, trace_flags) =
            parse_traceparent("01-11111111111111111111111111111111-2222222222222222-01");
        assert_eq!(
            trace_id.as_deref(),
            Some("11111111111111111111111111111111")
        );
        assert_eq!(trace_flags.as_deref(), Some("01"));
    }

    #[test]
    fn parse_traceparent_rejects_malformed() {
        // Wrong number of `-`-separated segments.
        assert_eq!(parse_traceparent("00-1111-2222"), (None, None, None)); // 3 segments
        assert_eq!(
            parse_traceparent("00-11111111111111111111111111111111-2222222222222222-00-extra"),
            (None, None, None)
        ); // 5 segments

        // All-or-nothing: any single invalid field rejects the whole parse.
        // (Per-field rules are covered by the is_valid_* tests above.)
        for tp in [
            "ff-11111111111111111111111111111111-2222222222222222-01", // bad version
            "00-bad-2222222222222222-01",                              // bad trace_id
            "00-11111111111111111111111111111111-bad-01",              // bad span_id
            "00-11111111111111111111111111111111-2222222222222222-0x", // bad flags
        ] {
            assert_eq!(
                parse_traceparent(tp),
                (None, None, None),
                "should reject: {tp}"
            );
        }
    }

    #[test]
    fn trace_parent_from_headers_preserves_unsampled_flag() {
        let mut headers = async_nats::HeaderMap::new();
        headers.insert(
            "traceparent",
            "00-11111111111111111111111111111111-2222222222222222-00",
        );

        let trace_parent = TraceParent::from_headers(&headers);

        assert_eq!(
            trace_parent.trace_id.as_deref(),
            Some("11111111111111111111111111111111")
        );
        assert_eq!(trace_parent.parent_id.as_deref(), Some("2222222222222222"));
        assert_eq!(trace_parent.trace_flags.as_deref(), Some("00"));
    }

    #[test]
    fn distributed_context_creates_traceparent_with_stored_flags() {
        let context = DistributedTraceContext {
            trace_id: "11111111111111111111111111111111".to_string(),
            span_id: "2222222222222222".to_string(),
            trace_flags: "00".to_string(),
            parent_id: None,
            tracestate: None,
            start: None,
            end: None,
            x_request_id: None,
            request_id: None,
        };

        assert_eq!(
            context.create_traceparent(),
            "00-11111111111111111111111111111111-2222222222222222-00"
        );
    }

    #[test]
    fn inject_trace_headers_preserves_current_span_flags() {
        // Use the core `set_default` (not `SubscriberInitExt::set_default`, which
        // also installs the global `log` LogTracer and would poison a later
        // `logging::init()` with SetLoggerError).
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(DistributedTraceIdLayer),
        );
        let span = tracing::info_span!(
            "root",
            trace_id = "11111111111111111111111111111111",
            span_id = "2222222222222222",
            trace_flags = "00"
        );
        let _enter = span.enter();
        let mut headers = std::collections::HashMap::new();

        inject_trace_headers_into_map(&mut headers);

        assert_eq!(
            headers.get("traceparent").map(String::as_str),
            Some("00-11111111111111111111111111111111-2222222222222222-00")
        );
    }

    #[test]
    fn request_span_preserves_inbound_trace_flags() {
        // Use the core `set_default` (not `SubscriberInitExt::set_default`, which
        // also installs the global `log` LogTracer and would poison a later
        // `logging::init()` with SetLoggerError).
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(DistributedTraceIdLayer),
        );
        let req = Request::builder()
            .header(
                "traceparent",
                "00-11111111111111111111111111111111-2222222222222222-00",
            )
            .body(())
            .unwrap();
        let trace_parent = TraceParent::from_headers(req.headers());
        let span = tracing::info_span!(
            "root",
            trace_id = trace_parent.trace_id,
            span_id = "3333333333333333",
            parent_id = trace_parent.parent_id,
            trace_flags = trace_parent.trace_flags
        );
        let _enter = span.enter();
        let mut headers = std::collections::HashMap::new();

        inject_trace_headers_into_map(&mut headers);

        assert_eq!(
            headers.get("traceparent").map(String::as_str),
            Some("00-11111111111111111111111111111111-3333333333333333-00")
        );
    }

    #[test]
    fn root_context_uses_otel_unsampled_decision() {
        let provider = SdkTracerProvider::builder()
            .with_sampler(opentelemetry_sdk::trace::Sampler::AlwaysOff)
            .build();
        let tracer = provider.tracer("test");
        // Core `set_default` (not `SubscriberInitExt::set_default`) to avoid
        // installing the global `log` LogTracer, which would poison a later
        // `logging::init()` with SetLoggerError.
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry()
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .with(DistributedTraceIdLayer),
        );
        let span = tracing::info_span!("root");
        let _enter = span.enter();
        let mut headers = std::collections::HashMap::new();

        inject_trace_headers_into_map(&mut headers);

        assert!(headers["traceparent"].ends_with("-00"));
        assert_eq!(
            get_distributed_tracing_context()
                .as_ref()
                .map(|ctx| ctx.trace_flags.as_str()),
            Some("00")
        );
    }

    #[test]
    fn root_context_uses_otel_sampled_decision() {
        let provider = SdkTracerProvider::builder()
            .with_sampler(opentelemetry_sdk::trace::Sampler::AlwaysOn)
            .build();
        let tracer = provider.tracer("test");
        // Core `set_default` (not `SubscriberInitExt::set_default`) to avoid
        // installing the global `log` LogTracer, which would poison a later
        // `logging::init()` with SetLoggerError.
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry()
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .with(DistributedTraceIdLayer),
        );
        let span = tracing::info_span!("root");
        let _enter = span.enter();
        let mut headers = std::collections::HashMap::new();

        inject_trace_headers_into_map(&mut headers);

        assert!(headers["traceparent"].ends_with("-01"));
        assert_eq!(
            get_distributed_tracing_context()
                .as_ref()
                .map(|ctx| ctx.trace_flags.as_str()),
            Some("01")
        );
    }

    #[tokio::test]
    async fn test_json_log_capture() -> Result<()> {
        #[allow(clippy::redundant_closure_call)]
        let _ = temp_env::async_with_vars(
            [(env_logging::DYN_LOGGING_JSONL, Some("1"))],
            (async || {
                let tmp_file = NamedTempFile::new().unwrap();
                let file_name = tmp_file.path().to_str().unwrap();
                let guard = StderrOverride::from_file(file_name)?;
                init();
                parent().await;
                drop(guard);

                let lines = load_log(file_name)?;

                // 1. Extract the dynamically generated trace ID and validate consistency
                // All logs should have the same trace_id since they're part of the same trace
                // Skip any initialization logs that don't have trace_id (e.g., OTLP setup messages)
                //
                // Note: This test can fail if logging was already initialized by another test running
                // in parallel. Logging initialization is global (Once) and can only happen once per process.
                // If no trace_id is found, skip validation gracefully.
                let Some(trace_id) = lines
                    .iter()
                    .find_map(|log_line| log_line.get("trace_id").and_then(|v| v.as_str()))
                    .map(|s| s.to_string())
                else {
                    // Skip test if logging was already initialized - we can't control the output format
                    return Ok(());
                };

                // Verify trace_id is not a zero/invalid ID
                assert_ne!(
                    trace_id, "00000000000000000000000000000000",
                    "trace_id should not be a zero/invalid ID"
                );
                assert!(
                    !trace_id.chars().all(|c| c == '0'),
                    "trace_id should not be all zeros"
                );

                // Verify all logs have the same trace_id
                for log_line in &lines {
                    if let Some(line_trace_id) = log_line.get("trace_id") {
                        assert_eq!(
                            line_trace_id.as_str().unwrap(),
                            &trace_id,
                            "All logs should have the same trace_id"
                        );
                    }
                }

                // Validate my_trace_id matches the actual trace ID
                for log_line in &lines {
                    if let Some(my_trace_id) = log_line.get("my_trace_id") {
                        assert_eq!(
                            my_trace_id,
                            &serde_json::Value::String(trace_id.clone()),
                            "my_trace_id should match the trace_id from distributed tracing context"
                        );
                    }
                }

                // 2. Validate span IDs exist and are properly formatted
                let mut span_ids_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                let mut span_timestamps: std::collections::HashMap<String, DateTime<Utc>> = std::collections::HashMap::new();

                for log_line in &lines {
                    if let Some(span_id) = log_line.get("span_id") {
                        let span_id_str = span_id.as_str().unwrap();
                        assert!(
                            is_valid_span_id(span_id_str),
                            "Invalid span_id format: {}",
                            span_id_str
                        );
                        span_ids_seen.insert(span_id_str.to_string());
                    }

                    // Validate timestamp format and track span timestamps
                    if let Some(time_str) = log_line.get("time").and_then(|v| v.as_str()) {
                        let timestamp = DateTime::parse_from_rfc3339(time_str)
                            .expect("All timestamps should be valid RFC3339 format")
                            .with_timezone(&Utc);

                        // Track timestamp for each span_name
                        if let Some(span_name) = log_line.get("span_name").and_then(|v| v.as_str()) {
                            span_timestamps.insert(span_name.to_string(), timestamp);
                        }
                    }
                }

                // 3. Validate parent-child span relationships
                // Extract span IDs for each span by looking at their log messages
                let parent_span_id = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("span_name")
                            .and_then(|v| v.as_str()) == Some("parent")
                    })
                    .and_then(|log_line| {
                        log_line.get("span_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .expect("Should find parent span with span_id");

                let child_span_id = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("span_name")
                            .and_then(|v| v.as_str()) == Some("child")
                    })
                    .and_then(|log_line| {
                        log_line.get("span_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .expect("Should find child span with span_id");

                let grandchild_span_id = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("span_name")
                            .and_then(|v| v.as_str()) == Some("grandchild")
                    })
                    .and_then(|log_line| {
                        log_line.get("span_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .expect("Should find grandchild span with span_id");

                // Verify span IDs are unique
                assert_ne!(parent_span_id, child_span_id, "Parent and child should have different span IDs");
                assert_ne!(child_span_id, grandchild_span_id, "Child and grandchild should have different span IDs");
                assert_ne!(parent_span_id, grandchild_span_id, "Parent and grandchild should have different span IDs");

                // Verify parent span has no parent_id
                for log_line in &lines {
                    if let Some(span_name) = log_line.get("span_name")
                        && let Some(span_name_str) = span_name.as_str()
                        && span_name_str == "parent"
                    {
                        assert!(
                            log_line.get("parent_id").is_none(),
                            "Parent span should not have a parent_id"
                        );
                    }
                }

                // Verify child span's parent_id is parent_span_id
                for log_line in &lines {
                    if let Some(span_name) = log_line.get("span_name")
                        && let Some(span_name_str) = span_name.as_str()
                        && span_name_str == "child"
                    {
                        let parent_id = log_line.get("parent_id")
                            .and_then(|v| v.as_str())
                            .expect("Child span should have a parent_id");
                        assert_eq!(
                            parent_id,
                            parent_span_id,
                            "Child's parent_id should match parent's span_id"
                        );
                    }
                }

                // Verify grandchild span's parent_id is child_span_id
                for log_line in &lines {
                    if let Some(span_name) = log_line.get("span_name")
                        && let Some(span_name_str) = span_name.as_str()
                        && span_name_str == "grandchild"
                    {
                        let parent_id = log_line.get("parent_id")
                            .and_then(|v| v.as_str())
                            .expect("Grandchild span should have a parent_id");
                        assert_eq!(
                            parent_id,
                            child_span_id,
                            "Grandchild's parent_id should match child's span_id"
                        );
                    }
                }

                // 4. Validate timestamp ordering - spans should log in execution order
                let parent_time = span_timestamps.get("parent")
                    .expect("Should have timestamp for parent span");
                let child_time = span_timestamps.get("child")
                    .expect("Should have timestamp for child span");
                let grandchild_time = span_timestamps.get("grandchild")
                    .expect("Should have timestamp for grandchild span");

                // Parent logs first (or at same time), then child, then grandchild
                assert!(
                    parent_time <= child_time,
                    "Parent span should log before or at same time as child span (parent: {}, child: {})",
                    parent_time,
                    child_time
                );
                assert!(
                    child_time <= grandchild_time,
                    "Child span should log before or at same time as grandchild span (child: {}, grandchild: {})",
                    child_time,
                    grandchild_time
                );

                Ok::<(), anyhow::Error>(())
            })(),
        )
        .await;
        Ok(())
    }

    #[test]
    fn test_otlp_export_works_without_json_logging() {
        use std::process::Command;

        let output = Command::new("cargo")
            .args([
                "test",
                "-p",
                "dynamo-runtime",
                "logging::tests::test_otlp_export_without_json_logging_subprocess",
                "--",
                "--exact",
                "--nocapture",
            ])
            .env("OTEL_EXPORT_ENABLED", "1")
            .env_remove("DYN_LOGGING_JSONL")
            .output()
            .expect("Failed to execute subprocess test");

        let stderr = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() {
            eprintln!(
                "=== STDOUT ===\n{}",
                String::from_utf8_lossy(&output.stdout)
            );
            eprintln!("=== STDERR ===\n{}", stderr);
        }

        assert!(
            output.status.success(),
            "Subprocess test failed with exit code: {:?}",
            output.status.code()
        );
        assert!(
            !stderr.contains("has no effect without DYN_LOGGING_JSONL"),
            "OTLP export should not depend on JSONL logging: {stderr}"
        );
        assert!(
            stderr.contains("OpenTelemetry OTLP export enabled"),
            "OTLP export should initialize with readable logging: {stderr}"
        );
    }

    #[tokio::test]
    async fn test_otlp_export_without_json_logging_subprocess() {
        if std::env::var("OTEL_EXPORT_ENABLED").is_err() {
            return;
        }

        init();
        tracing::info!("readable log with OTLP export");
    }

    // Test functions at different log levels for filtering tests
    #[tracing::instrument(level = "debug", skip_all)]
    async fn debug_level_span() {
        tracing::debug!("inside debug span");
    }

    #[tracing::instrument(level = "info", skip_all)]
    async fn info_level_span() {
        tracing::info!("inside info span");
    }

    #[tracing::instrument(level = "warn", skip_all)]
    async fn warn_level_span() {
        tracing::warn!("inside warn span");
    }

    // Span from a different target - should be FILTERED OUT at info level
    // because the filter is warn,dynamo_runtime::logging::tests=debug
    #[tracing::instrument(level = "info", target = "other_module", skip_all)]
    async fn other_target_info_span() {
        tracing::info!(target: "other_module", "inside other target span");
    }

    /// Comprehensive test for span events covering:
    /// - SPAN_FIRST_ENTRY and SPAN_CLOSED event emission
    /// - Trace context (trace_id, span_id) in span events
    /// - Timing information in SPAN_CLOSED events
    /// - Level-based filtering (positive: allowed levels pass, negative: filtered levels blocked)
    /// - Target-based filtering (spans from allowed targets pass even at lower levels)
    ///
    /// This test runs in a subprocess to ensure logging is initialized with our specific
    /// filter settings (DYN_LOG=warn,dynamo_runtime::logging::tests=debug), avoiding
    /// interference from other tests that may have initialized logging first.
    #[test]
    fn test_span_events() {
        use std::process::Command;

        // Run cargo test for the subprocess test with specific env vars
        let output = Command::new("cargo")
            .args([
                "test",
                "-p",
                "dynamo-runtime",
                "test_span_events_subprocess",
                "--",
                "--exact",
                "--nocapture",
            ])
            .env("DYN_LOGGING_JSONL", "1")
            .env("DYN_LOGGING_SPAN_EVENTS", "1")
            .env("DYN_LOG", "warn,dynamo_runtime::logging::tests=debug")
            .output()
            .expect("Failed to execute subprocess test");

        // Print output for debugging
        if !output.status.success() {
            eprintln!(
                "=== STDOUT ===\n{}",
                String::from_utf8_lossy(&output.stdout)
            );
            eprintln!(
                "=== STDERR ===\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        assert!(
            output.status.success(),
            "Subprocess test failed with exit code: {:?}",
            output.status.code()
        );
    }

    /// Subprocess test that performs the actual span event validation.
    /// This is called by test_span_events in a separate process with controlled env vars.
    #[tokio::test]
    async fn test_span_events_subprocess() -> Result<()> {
        // Skip if not running as subprocess (env vars not set)
        if std::env::var("DYN_LOGGING_SPAN_EVENTS").is_err() {
            return Ok(());
        }

        let tmp_file = NamedTempFile::new().unwrap();
        let file_name = tmp_file.path().to_str().unwrap();
        let guard = StderrOverride::from_file(file_name)?;
        init();

        // Run parent/child/grandchild spans (all INFO level by default)
        parent().await;

        // Run spans at explicit levels from our test module
        debug_level_span().await;
        info_level_span().await;
        warn_level_span().await;

        // Run span from different target (should be filtered out)
        other_target_info_span().await;

        drop(guard);

        let lines = load_log(file_name)?;

        // Helper to check if a span event exists
        let has_span_event = |msg: &str, span_name: &str| {
            lines.iter().any(|log| {
                log.get("message").and_then(|v| v.as_str()) == Some(msg)
                    && log.get("span_name").and_then(|v| v.as_str()) == Some(span_name)
            })
        };

        // Helper to get span events
        let get_span_events = |msg: &str| -> Vec<&serde_json::Value> {
            lines
                .iter()
                .filter(|log| log.get("message").and_then(|v| v.as_str()) == Some(msg))
                .collect()
        };

        // === Test 1: SPAN_FIRST_ENTRY events have required fields ===
        let span_created_events = get_span_events("SPAN_FIRST_ENTRY");
        for event in &span_created_events {
            // Must have span_name
            assert!(
                event.get("span_name").is_some(),
                "SPAN_FIRST_ENTRY must have span_name"
            );
            // Must have valid trace_id (format check)
            let trace_id = event
                .get("trace_id")
                .and_then(|v| v.as_str())
                .expect("SPAN_FIRST_ENTRY must have trace_id");
            assert!(
                trace_id.len() == 32 && trace_id.chars().all(|c| c.is_ascii_hexdigit()),
                "SPAN_FIRST_ENTRY must have valid trace_id format"
            );
            // Must have valid span_id
            let span_id = event
                .get("span_id")
                .and_then(|v| v.as_str())
                .expect("SPAN_FIRST_ENTRY must have span_id");
            assert!(
                is_valid_span_id(span_id),
                "SPAN_FIRST_ENTRY must have valid span_id"
            );
        }

        // === Test 2: SPAN_CLOSED events have timing info ===
        let span_closed_events = get_span_events("SPAN_CLOSED");
        for event in &span_closed_events {
            assert!(
                event.get("span_name").is_some(),
                "SPAN_CLOSED must have span_name"
            );
            assert!(
                event.get("time.busy_us").is_some()
                    || event.get("time.idle_us").is_some()
                    || event.get("time.duration_us").is_some(),
                "SPAN_CLOSED must have timing information"
            );
            // Must have valid trace_id
            let trace_id = event
                .get("trace_id")
                .and_then(|v| v.as_str())
                .expect("SPAN_CLOSED must have trace_id");
            assert!(
                trace_id.len() == 32 && trace_id.chars().all(|c| c.is_ascii_hexdigit()),
                "SPAN_CLOSED must have valid trace_id format"
            );
        }

        // === Test 3: Target-based filtering (positive) ===
        // Spans from dynamo_runtime::logging::tests should pass at ALL levels
        // because the target is allowed at debug level
        assert!(
            has_span_event("SPAN_FIRST_ENTRY", "debug_level_span"),
            "DEBUG span from allowed target MUST pass (target=debug filter)"
        );
        assert!(
            has_span_event("SPAN_FIRST_ENTRY", "info_level_span"),
            "INFO span from allowed target MUST pass (target=debug filter)"
        );
        assert!(
            has_span_event("SPAN_FIRST_ENTRY", "warn_level_span"),
            "WARN span from allowed target MUST pass (target=debug filter)"
        );

        // parent/child/grandchild are INFO level from allowed target - should pass
        assert!(
            has_span_event("SPAN_FIRST_ENTRY", "parent"),
            "parent span (INFO) from allowed target MUST pass"
        );
        assert!(
            has_span_event("SPAN_FIRST_ENTRY", "child"),
            "child span (INFO) from allowed target MUST pass"
        );
        assert!(
            has_span_event("SPAN_FIRST_ENTRY", "grandchild"),
            "grandchild span (INFO) from allowed target MUST pass"
        );

        // === Test 4: Level-based filtering (negative) ===
        // Verify spans from OTHER targets at debug/info level are filtered out
        assert!(
            !has_span_event("SPAN_FIRST_ENTRY", "other_target_info_span"),
            "INFO span from non-allowed target (other_module) MUST be filtered out"
        );

        // Also verify no spans from other targets appear at debug/info level
        for event in &span_created_events {
            let target = event.get("target").and_then(|v| v.as_str()).unwrap_or("");
            let level = event.get("level").and_then(|v| v.as_str()).unwrap_or("");

            // If level is DEBUG or INFO, target must be our test module
            if level == "DEBUG" || level == "INFO" {
                assert!(
                    target.contains("dynamo_runtime::logging::tests"),
                    "DEBUG/INFO span must be from allowed target, got target={target}"
                );
            }
        }

        Ok(())
    }
}
