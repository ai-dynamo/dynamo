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

use crate::config::{disable_ansi_logging, jsonl_logging_enabled};
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
use opentelemetry::trace::TraceContextExt;
use opentelemetry::{global, trace::Tracer};
use opentelemetry_otlp::WithExportConfig;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{Key, KeyValue};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing::error;
use tracing_subscriber::layer::SubscriberExt;
// use tracing_subscriber::Registry;

use std::time::Duration;
use tracing::{info, instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::config::environment_names::logging as env_logging;

use dynamo_config::env_is_truthy;

/// Default log level
const DEFAULT_FILTER_LEVEL: &str = "info";

/// Default OTLP endpoint
const DEFAULT_OTLP_ENDPOINT: &str = "http://localhost:4317";

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
                ("mistralrs_core".to_string(), "error".to_string()),
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
    pub x_dynamo_request_id: Option<String>,
}

/// Pending context data collected in on_new_span, to be finalized in on_enter
#[derive(Debug, Clone)]
struct PendingDistributedTraceContext {
    trace_id: Option<String>,
    span_id: Option<String>,
    parent_id: Option<String>,
    tracestate: Option<String>,
    x_request_id: Option<String>,
    x_dynamo_request_id: Option<String>,
}

impl DistributedTraceContext {
    /// Create a traceparent string from the context
    pub fn create_traceparent(&self) -> String {
        format!("00-{}-{}-01", self.trace_id, self.span_id)
    }
}

/// Parse a traceparent string into its components
pub fn parse_traceparent(traceparent: &str) -> (Option<String>, Option<String>) {
    let pieces: Vec<_> = traceparent.split('-').collect();
    if pieces.len() != 4 {
        return (None, None);
    }
    let trace_id = pieces[1];
    let parent_id = pieces[2];

    if !is_valid_trace_id(trace_id) || !is_valid_span_id(parent_id) {
        return (None, None);
    }

    (Some(trace_id.to_string()), Some(parent_id.to_string()))
}

#[derive(Debug, Clone, Default)]
pub struct TraceParent {
    pub trace_id: Option<String>,
    pub parent_id: Option<String>,
    pub tracestate: Option<String>,
    pub x_request_id: Option<String>,
    pub x_dynamo_request_id: Option<String>,
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
        let mut tracestate = None;
        let mut x_request_id = None;
        let mut x_dynamo_request_id = None;

        if let Some(header_value) = headers.get("traceparent") {
            (trace_id, parent_id) = parse_traceparent(header_value);
        }

        if let Some(header_value) = headers.get("x-request-id") {
            x_request_id = Some(header_value.to_string());
        }

        if let Some(header_value) = headers.get("tracestate") {
            tracestate = Some(header_value.to_string());
        }

        if let Some(header_value) = headers.get("x-dynamo-request-id") {
            x_dynamo_request_id = Some(header_value.to_string());
        }

        // Validate UUID format
        let x_dynamo_request_id =
            x_dynamo_request_id.filter(|id| uuid::Uuid::parse_str(id).is_ok());
        TraceParent {
            trace_id,
            parent_id,
            tracestate,
            x_request_id,
            x_dynamo_request_id,
        }
    }
}

// Takes Axum request and returning a span
pub fn make_request_span<B>(req: &Request<B>) -> Span {
    let method = req.method();
    let uri = req.uri();
    let version = format!("{:?}", req.version());
    let trace_parent = TraceParent::from_headers(req.headers());

    let span = tracing::info_span!(
        "http-request",
        method = %method,
        uri = %uri,
        version = %version,
        trace_id = trace_parent.trace_id,
        parent_id = trace_parent.parent_id,
        x_request_id = trace_parent.x_request_id,
    x_dynamo_request_id = trace_parent.x_dynamo_request_id,
    );

    span
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
            "handle_payload",
            trace_id = trace_id.as_str(),
            parent_id = parent_id.as_str(),
            x_request_id = trace_parent.x_request_id,
            x_dynamo_request_id = trace_parent.x_dynamo_request_id,
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
            "handle_payload",
            x_request_id = trace_parent.x_request_id,
            x_dynamo_request_id = trace_parent.x_dynamo_request_id,
            tracestate = trace_parent.tracestate,
            component = component,
            endpoint = endpoint,
            namespace = namespace,
            instance_id = instance_id,
        )
    }
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

    let (trace_id, parent_span_id) = parse_traceparent(traceparent_value);

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
    let propagator = opentelemetry_sdk::propagation::TraceContextPropagator::new();
    let otel_context = propagator.extract(&extractor);

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
    let propagator = opentelemetry_sdk::propagation::TraceContextPropagator::new();
    propagator.inject_context(&otel_context, &mut injector);
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
        if let Some(x_dynamo_request_id) = trace_context.x_dynamo_request_id {
            headers.insert("x-dynamo-request-id".to_string(), x_dynamo_request_id);
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
                x_request_id = ctx.x_request_id.as_deref(),
                x_dynamo_request_id = ctx.x_dynamo_request_id.as_deref(),
                // tracestate = ctx.tracestate.as_deref(),
            )
        } else {
            tracing::info_span!(
                "client_request",
                operation = operation,
                request_id = request_id,
                trace_id = ctx.trace_id.as_str(),
                parent_id = ctx.span_id.as_str(),
                x_request_id = ctx.x_request_id.as_deref(),
                x_dynamo_request_id = ctx.x_dynamo_request_id.as_deref(),
                // tracestate = ctx.tracestate.as_deref(),
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
            let mut x_request_id: Option<String> = None;
            let mut x_dynamo_request_id: Option<String> = None;
            let mut tracestate: Option<String> = None;
            let mut visitor = FieldVisitor::default();
            attrs.record(&mut visitor);

            // Extract trace_id from span attributes
            if let Some(trace_id_input) = visitor.fields.get("trace_id") {
                if !is_valid_trace_id(trace_id_input) {
                    tracing::trace!("trace id  '{}' is not valid! Ignoring.", trace_id_input);
                } else {
                    trace_id = Some(trace_id_input.to_string());
                }
            }

            // Extract span_id from span attributes
            if let Some(span_id_input) = visitor.fields.get("span_id") {
                if !is_valid_span_id(span_id_input) {
                    tracing::trace!("span id  '{}' is not valid! Ignoring.", span_id_input);
                } else {
                    span_id = Some(span_id_input.to_string());
                }
            }

            // Extract parent_id from span attributes
            if let Some(parent_id_input) = visitor.fields.get("parent_id") {
                if !is_valid_span_id(parent_id_input) {
                    tracing::trace!("parent id  '{}' is not valid! Ignoring.", parent_id_input);
                } else {
                    parent_id = Some(parent_id_input.to_string());
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

            // Extract x_dynamo_request_id
            if let Some(x_request_id_input) = visitor.fields.get("x_dynamo_request_id") {
                x_dynamo_request_id = Some(x_request_id_input.to_string());
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
                    tracestate = parent_tracing_context.tracestate.clone();
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
                tracestate,
                x_request_id,
                x_dynamo_request_id,
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
            let tracestate = pending.tracestate;
            let x_request_id = pending.x_request_id;
            let x_dynamo_request_id = pending.x_dynamo_request_id;

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

            // Panic if we still don't have required IDs
            if trace_id.is_none() {
                panic!(
                    "trace_id is not set in on_enter - OtelData may not be properly initialized"
                );
            }

            if span_id.is_none() {
                panic!("span_id is not set in on_enter - OtelData may not be properly initialized");
            }

            // Re-acquire mutable borrow to insert the finalized context
            let mut extensions = span.extensions_mut();
            extensions.insert(DistributedTraceContext {
                trace_id: trace_id.expect("Trace ID must be set"),
                span_id: span_id.expect("Span ID must be set"),
                parent_id,
                tracestate,
                start: Some(Instant::now()),
                end: None,
                x_request_id,
                x_dynamo_request_id,
            });
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
fn setup_logging() {
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
}

#[cfg(not(feature = "tokio-console"))]
fn setup_logging() -> Result<(), Box<dyn std::error::Error>> {
    let fmt_filter_layer = filters(load_config());
    let trace_filter_layer = filters(load_config());
    let otel_filter_layer = filters(load_config());

    if jsonl_logging_enabled() {
        let l = fmt::layer()
            .with_ansi(false)
            .event_format(CustomJsonFormatter::new())
            .with_writer(std::io::stderr)
            .with_filter(fmt_filter_layer);

        // Create OpenTelemetry tracer - conditionally export to OTLP based on env var
        let service_name = get_service_name();

        // Build tracer provider - with or without OTLP export
        let (tracer_provider, endpoint_opt) = if otlp_exporter_enabled() {
            // Export enabled: create OTLP exporter with batch processor
            let endpoint = std::env::var(env_logging::otlp::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)
                .unwrap_or_else(|_| DEFAULT_OTLP_ENDPOINT.to_string());

            // Initialize OTLP exporter using gRPC (Tonic)
            let otlp_exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .with_endpoint(&endpoint)
                .build()?;

            // Create tracer provider with batch exporter and service name
            let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_batch_exporter(otlp_exporter)
                .with_resource(
                    opentelemetry_sdk::Resource::builder_empty()
                        .with_service_name(service_name.clone())
                        .build(),
                )
                .build();

            (provider, Some(endpoint))
        } else {
            // No export - traces generated locally only (for logging/trace IDs)
            let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
                .with_resource(
                    opentelemetry_sdk::Resource::builder_empty()
                        .with_service_name(service_name.clone())
                        .build(),
                )
                .build();

            (provider, None)
        };

        // Get a tracer from the provider
        let tracer = tracer_provider.tracer(service_name.clone());

        tracing_subscriber::registry()
            .with(
                tracing_opentelemetry::layer()
                    .with_tracer(tracer)
                    .with_filter(otel_filter_layer),
            )
            .with(DistributedTraceIdLayer.with_filter(trace_filter_layer))
            .with(l)
            .init();

        // Log initialization status after subscriber is ready
        if let Some(endpoint) = endpoint_opt {
            tracing::info!(
                endpoint = %endpoint,
                service = %service_name,
                "OpenTelemetry OTLP export enabled"
            );
        } else {
            tracing::info!(
                service = %service_name,
                "OpenTelemetry OTLP export disabled, traces local only"
            );
        }
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
    target: &'a str,
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

            message = match message.as_str() {
                Some("new") => serde_json::Value::String("SPAN_CREATED".to_string()),
                Some("close") => serde_json::Value::String("SPAN_CLOSED".to_string()),
                _ => message.clone(),
            };

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

                if let Some(x_dynamo_request_id) = tracing_context.x_dynamo_request_id.clone() {
                    visitor.fields.insert(
                        "x_dynamo_request_id".to_string(),
                        serde_json::Value::String(x_dynamo_request_id),
                    );
                } else {
                    visitor.fields.remove("x_dynamo_request_id");
                }
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
            target: metadata.target(),
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
}
