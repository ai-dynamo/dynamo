// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
use std::sync::atomic::{AtomicBool, Ordering};

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

// // OpenTelemetry imports
// use opentelemetry::global;
// use opentelemetry::trace::{TraceContextExt};
// use opentelemetry::propagation::{Extractor, Injector, TextMapPropagator};
// use opentelemetry_otlp::WithExportConfig;
// use tracing_opentelemetry::OpenTelemetrySpanExt;


use opentelemetry::global;
use opentelemetry::trace::{TraceContextExt};
use opentelemetry::propagation::{Extractor, Injector, TextMapPropagator};
use opentelemetry_otlp::WithExportConfig;
// use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing_opentelemetry::OpenTelemetrySpanExt;
// use std::fmt;


/// ENV used to set the log level
const FILTER_ENV: &str = "DYN_LOG";

/// Default log level
const DEFAULT_FILTER_LEVEL: &str = "info";

/// ENV used to set the path to the logging configuration file
const CONFIG_PATH_ENV: &str = "DYN_LOGGING_CONFIG_PATH";

/// Once instance to ensure the logger is only initialized once
static INIT: Once = Once::new();

/// Global flag to track if lazy OTEL initialization has been completed
static LAZY_OTEL_INITIALIZED: AtomicBool = AtomicBool::new(false);

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
            ]),
        }
    }
}

/// Generate a 32-character, lowercase hex trace ID (W3C-compliant)
fn generate_trace_id() -> String {
    Uuid::new_v4().simple().to_string()
}

/// Generate a 16-character, lowercase hex span ID (W3C-compliant)
fn generate_span_id() -> String {
    // Use the first 8 bytes (16 hex chars) of a UUID v4
    let uuid = Uuid::new_v4();
    let bytes = uuid.as_bytes();
    bytes[..8].iter().map(|b| format!("{:02x}", b)).collect()
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

        // TODO: these should not be extracted from the headers but rather taken from the extractor/propagator
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
// TODO: This is the entrypoint for the request
// Create a new span for the request
// We want to check if the headers container a parent context by calling extract_trace_context_from_headers
// if there is a parent context, we call set_parent on the span
// extract the trace id from the .context() and set it as a field on the span (we'll do some work to persist this in the extensions of the span)
// Note: This is only one of the places

// TODO: Create a new span for the request
// If there is a parent context (this is part of a different request, call extract_trace_context_from_headers and set as parent). This will require implementing the extractor for http headers
// Get trace id, write this as a field of the trace_parent
// propagator is still needed to set the trace id and span id in ways that otel can use it and propagate it
// IMPROVEMENT: figure out if we can use the propagator for all keys
// (distributed_span_from_headers):This is a more general function that takes in headers, determines if there is a parent context within them, and returns a span with the parent context correctly configured passing in the trace id as needed
// as a first pass, we can potentially have a subroutine that intializes a traceparent as well, some of the fields are read from the headers and some are read from otel extension
// we can later figure out how to ship all the details in the otel context
pub fn make_request_span<B>(req: &Request<B>) -> Span {
    // Attempt lazy OTEL initialization on first request
    lazy_init_otel();
    
    let method = req.method();
    let uri = req.uri();
    let version = format!("{:?}", req.version());

    // porentially we can preserve this API and make it a bridge to the propagation system
    let trace_parent = TraceParent::from_headers(req.headers());

    // TODO: Remove later.
    // Useful to verify that everything should be None, since this is the first span of the request
    assert!(trace_parent.trace_id.is_none());
    assert!(trace_parent.parent_id.is_none());
    assert!(trace_parent.tracestate.is_none());
    assert!(trace_parent.x_request_id.is_none());
    assert!(trace_parent.x_dynamo_request_id.is_none());

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


/// Create a span from NATS headers representing an entry point across a process boundary
/// 
/// This function extracts trace context from incoming NATS headers and creates a new span
/// that properly continues the distributed trace. The span will have the extracted context
/// as its parent and will include trace_id and parent_id fields for bootstrapping the
/// distributed tracing system.
/// 
/// # Arguments
/// * `headers` - Reference to async_nats::HeaderMap containing trace context headers
/// 
/// # Returns
/// A configured tracing::Span that represents the entry point for this service
/// 
/// # Example
/// ```rust
/// let span = create_span_from_nats_headers(&headers);
/// let _enter = span.enter();
/// // All subsequent spans will be children of this entry span
/// ```
pub fn create_span_from_nats_headers(
    headers: &async_nats::HeaderMap,
) -> Span {
    // Extract the trace context and individual IDs from headers
    let (otel_context, trace_id, parent_span_id) = extract_otel_context_from_nats_headers(headers);
    
    // Parse the TraceParent struct to get all additional fields
    let trace_parent = TraceParent::from_headers(headers);
    
    // Create the span with trace context fields for bootstrapping
    // Use span! macro with target and level to allow dynamic span names
    let span = if let (Some(trace_id), Some(parent_id)) = (trace_id.as_ref(), parent_span_id.as_ref()) {
        // Create span with trace_id and parent_id fields to bootstrap our distributed tracing layer
        // Include all additional fields from the TraceParent struct
        let span = tracing::info_span!(
            "continuing-request-across-process-boundary",
            trace_id = trace_id.as_str(),
            parent_id = parent_id.as_str(),
            x_request_id = trace_parent.x_request_id,
            x_dynamo_request_id = trace_parent.x_dynamo_request_id,
            tracestate = trace_parent.tracestate,
        );

        // Set the extracted OpenTelemetry context as the parent if available
        if let Some(context) = otel_context {
            span.set_parent(context);
        }
        span
    } else {
        // No trace context found, create a regular span (this will generate new trace context)
        // Still include additional fields from TraceParent if they exist
        let span = tracing::info_span!(
            "new-request",
            x_request_id = trace_parent.x_request_id,
            x_dynamo_request_id = trace_parent.x_dynamo_request_id,
            tracestate = trace_parent.tracestate,
        );
        span
    };
    
    span
}

/// Create a handle_payload span from NATS headers with additional context fields
/// 
/// This function creates a span specifically for handling payloads in the push endpoint.
/// It extracts trace context from NATS headers and includes component, endpoint, namespace,
/// and instance_id fields for better observability.
/// 
/// # Arguments
/// * `headers` - Reference to async_nats::HeaderMap containing trace context headers
/// * `component` - Component name to include in the span
/// * `endpoint` - Endpoint name to include in the span
/// * `namespace` - Namespace to include in the span
/// * `instance_id` - Instance ID to include in the span
/// 
/// # Returns
/// A configured tracing::Span for handling payloads
pub fn create_handle_payload_span_from_nats_headers(
    headers: &async_nats::HeaderMap,
    component: &str,
    endpoint: &str,
    namespace: &str,
    instance_id: i64,
) -> Span {
    // Attempt lazy OTEL initialization on NATS message handling
    lazy_init_otel();
    
    // Extract the trace context and individual IDs from headers
    let (otel_context, trace_id, parent_span_id) = extract_otel_context_from_nats_headers(headers);
    
    // Parse the TraceParent struct to get all additional fields
    let trace_parent = TraceParent::from_headers(headers);
    
    // Create the span with trace context fields for bootstrapping
    let span = if let (Some(trace_id), Some(parent_id)) = (trace_id.as_ref(), parent_span_id.as_ref()) {
        // Create span with trace_id and parent_id fields to bootstrap our distributed tracing layer
        // Include all additional fields from the TraceParent struct and context fields
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

        // Set the extracted OpenTelemetry context as the parent if available
        if let Some(context) = otel_context {
            span.set_parent(context);
        }
        span
    } else {
        // No trace context found, create a regular span (this will generate new trace context)
        // Still include additional fields from TraceParent if they exist
        let span = tracing::info_span!(
            "handle_payload",
            x_request_id = trace_parent.x_request_id,
            x_dynamo_request_id = trace_parent.x_dynamo_request_id,
            tracestate = trace_parent.tracestate,
            component = component,
            endpoint = endpoint,
            namespace = namespace,
            instance_id = instance_id,
        );
        span
    };
    
    span
}

/// Extract OpenTelemetry trace context from async_nats::HeaderMap
/// 
/// This function uses the OpenTelemetry extractor pattern to extract trace context
/// from NATS headers for distributed tracing across service boundaries.
/// 
/// # Arguments
/// * `headers` - Reference to async_nats::HeaderMap to extract context from
/// 
/// # Returns
/// A tuple containing:
/// * `Option<opentelemetry::Context>` - The extracted OTEL context (None if no traceparent found)
/// * `Option<String>` - The trace_id from the traceparent header
/// * `Option<String>` - The span_id from the traceparent header (this becomes the parent_id for child spans)
pub fn extract_otel_context_from_nats_headers(
    headers: &async_nats::HeaderMap,
) -> (Option<opentelemetry::Context>, Option<String>, Option<String>) {
    // First check if traceparent header exists
    let traceparent_value = match headers.get("traceparent") {
        Some(value) => value.as_str(),
        None => {
            // No traceparent header found, return None for all values
            return (None, None, None);
        }
    };

    // Parse the traceparent header to extract trace_id and span_id
    let (trace_id, parent_span_id) = parse_traceparent(traceparent_value);

    // Create an extractor that reads from the NATS headers map
    struct NatsHeaderExtractor<'a>(&'a async_nats::HeaderMap);

    impl<'a> Extractor for NatsHeaderExtractor<'a> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).map(|value| value.as_str())
        }

        fn keys(&self) -> Vec<&str> {
            // Convert HeaderMap keys to a Vec of string references
            // Note: async_nats::HeaderMap doesn't expose all keys directly,
            // but the propagator typically only looks for known keys like "traceparent" and "tracestate"
            vec!["traceparent", "tracestate"]
                .into_iter()
                .filter(|&key| self.0.get(key).is_some())
                .collect()
        }
    }

    let extractor = NatsHeaderExtractor(headers);
    let propagator = opentelemetry::sdk::propagation::TraceContextPropagator::new();
    
    // Extract the context from headers using the W3C Trace Context propagation standard
    let otel_context = propagator.extract(&extractor);

    // Check if we actually got a valid context (not just the default empty context)
    let context_with_trace = if otel_context.span().span_context().is_valid() {
        Some(otel_context)
    } else {
        None
    };

    (context_with_trace, trace_id, parent_span_id)
}


// TODO: prepare_headers_from_span: method that takes the current span and prepares the headers for the outbound request (writing trace_id and span_id) to continue the distributed trace
// just call the injector pattern after getting the context from the current span
// improv: pass additional headers
/// Simulate injecting trace context into HTTP headers (outgoing request)
// fn inject_trace_context_into_headers(context: &opentelemetry::Context) -> HashMap<String, String> {
//     let mut headers = HashMap::new();
    
//     // Create an injector that writes to our headers map
//     struct HeaderInjector<'a>(&'a mut HashMap<String, String>);
    
//     impl<'a> Injector for HeaderInjector<'a> {
//         fn set(&mut self, key: &str, value: String) {
//             self.0.insert(key.to_string(), value);
//         }
//     }

//     let mut injector = HeaderInjector(&mut headers);
//     let propagator = opentelemetry::sdk::propagation::TraceContextPropagator::new();
    
//     // Inject the current context into headers
//     propagator.inject_context(context, &mut injector);
    
//     headers
// }
/// Inject OpenTelemetry trace context into async_nats::HeaderMap
/// 
/// This function uses the OpenTelemetry propagator pattern to inject trace context
/// into NATS headers for distributed tracing across service boundaries.
/// 
/// # Arguments
/// * `headers` - Mutable reference to async_nats::HeaderMap to inject headers into
/// * `context` - Optional OpenTelemetry context. If None, derives from current span
pub fn inject_otel_context_into_nats_headers(
    headers: &mut async_nats::HeaderMap,
    context: Option<opentelemetry::Context>,
) {
    // Get the context from parameter or derive from current span
    let otel_context = context.unwrap_or_else(|| {
        let current_span = Span::current();
        current_span.context()
    });

    // Create an injector that writes to the NATS headers map
    struct NatsHeaderInjector<'a>(&'a mut async_nats::HeaderMap);

    impl<'a> Injector for NatsHeaderInjector<'a> {
        fn set(&mut self, key: &str, value: String) {
            self.0.insert(key, value);
        }
    }

    let mut injector = NatsHeaderInjector(headers);
    let propagator = opentelemetry::sdk::propagation::TraceContextPropagator::new();
    
    // Inject the context into headers using the W3C Trace Context propagation standard
    propagator.inject_context(&otel_context, &mut injector);
}

/// Convenience function to inject trace context from current span into NATS headers
/// 
/// This is a simplified version that automatically uses the current span's context.
pub fn inject_current_trace_into_nats_headers(headers: &mut async_nats::HeaderMap) {
    inject_otel_context_into_nats_headers(headers, None);
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
        println!("on_close called for span");
        if let Some(span) = ctx.span(&id) {
            let mut extensions = span.extensions_mut();
            if let Some(distributed_tracing_context) =
                extensions.get_mut::<DistributedTraceContext>()
            {
                distributed_tracing_context.end = Some(Instant::now());
            }
        }
    }

    // Adds W3C compliant span_id, trace_id, and parent_id if not already present
    // TODO: Change to on_enter
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        println!("on_new_span");
        // Handling of trace_id:
        // 1. Check if passed in (first span across a network boundary)
        // 2. Check if set on parent span (all spans not first across a network boundary)
        // 3. trace_id is retrieved from the builder (first span in a request)

        // span id on the other hand can always be reliably retrieved from the otel extensions
        if let Some(span) = ctx.span(id) {
            let mut trace_id: Option<String> = None;
            let mut parent_id: Option<String> = None;
            let mut span_id: Option<String> = None;
            let mut x_request_id: Option<String> = None;
            let mut x_dynamo_request_id: Option<String> = None;
            let mut tracestate: Option<String> = None;
            let mut visitor = FieldVisitor::default();
            attrs.record(&mut visitor);

            if let Some(trace_id_input) = visitor.fields.get("trace_id") {
                if !is_valid_trace_id(trace_id_input) {
                    tracing::trace!("trace id  '{}' is not valid! Ignoring.", trace_id_input);
                } else {
                    println!("trace id resolved from fields: {}", trace_id_input);
                    trace_id = Some(trace_id_input.to_string());
                }
            }

            if let Some(span_id_input) = visitor.fields.get("span_id") {
                if !is_valid_span_id(span_id_input) {
                    tracing::trace!("span id  '{}' is not valid! Ignoring.", span_id_input);
                } else {
                    span_id = Some(span_id_input.to_string());
                }
            }

            if let Some(parent_id_input) = visitor.fields.get("parent_id") {
                if !is_valid_span_id(parent_id_input) {
                    tracing::trace!("parent id  '{}' is not valid! Ignoring.", parent_id_input);
                } else {
                    parent_id = Some(parent_id_input.to_string());
                }
            }

            if let Some(tracestate_input) = visitor.fields.get("tracestate") {
                tracestate = Some(tracestate_input.to_string());
            }

            if let Some(x_request_id_input) = visitor.fields.get("x_request_id") {
                x_request_id = Some(x_request_id_input.to_string());
            }

            if let Some(x_request_id_input) = visitor.fields.get("x_dynamo_request_id") {
                x_dynamo_request_id = Some(x_request_id_input.to_string());
            }

            // TODO: verify that set_parent doesn't mean a parent span is set
            if parent_id.is_none()
                && let Some(parent_span_id) = ctx.current_span().id()
                && let Some(parent_span) = ctx.span(parent_span_id)
            {
                let parent_ext = parent_span.extensions();
                if let Some(parent_tracing_context) = parent_ext.get::<DistributedTraceContext>() {
                    println!("trace id resolved from parent span: {}", parent_tracing_context.trace_id);
                    trace_id = Some(parent_tracing_context.trace_id.clone());
                    parent_id = Some(parent_tracing_context.span_id.clone());
                    tracestate = parent_tracing_context.tracestate.clone();
                }
            }

            if (parent_id.is_some() || span_id.is_some()) && trace_id.is_none() {
                tracing::error!("parent id or span id are set but trace id is not set!");
                // Clear inconsistent IDs to maintain trace integrity
                parent_id = None;
                span_id = None;
            }

            // Extract trace_id and span_id from OpenTelemetry extensions if not already set
            if trace_id.is_none() || span_id.is_none() {
                let extensions = span.extensions();
                if let Some(otel_data) = extensions.get::<tracing_opentelemetry::OtelData>() {
                    // Extract trace_id from OTEL data if not already set
                    if trace_id.is_none() {
                        if let Some(otel_trace_id) = otel_data.builder.trace_id {
                            let trace_id_str = format!("{}", otel_trace_id);
                            if is_valid_trace_id(&trace_id_str) {
                                println!("trace id resolved from OTEL data: {}", trace_id_str);
                                trace_id = Some(trace_id_str);
                            } else {
                                println!("Invalid trace_id from OTEL data: {}", trace_id_str);
                            }
                        }
                    }
                    
                    // Extract span_id from OTEL data if not already set
                    if span_id.is_none() {
                        if let Some(otel_span_id) = otel_data.builder.span_id {
                            let span_id_str = format!("{}", otel_span_id);
                            if is_valid_span_id(&span_id_str) {
                                span_id = Some(span_id_str);
                            } else {
                                println!("Invalid span_id from OTEL data: {}", span_id_str);
                            }
                        }
                    }
                } else {
                    println!("No OpenTelemetry data found in span extensions");
                }
            }

            if trace_id.is_none() {
                println!("trace id is not set");
                panic!("trace id is not set");
            }

            if span_id.is_none() {
                println!("span id is not set");
                panic!("span id is not set");
            }

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

/// Lazy initialize OTEL tracing when Tokio runtime is available
/// This should be called from request processing functions
pub fn lazy_init_otel() {
    // Only attempt lazy init if JSONL logging is enabled and not already initialized
    if !jsonl_logging_enabled() || LAZY_OTEL_INITIALIZED.load(Ordering::Relaxed) {
        return;
    }
    
    // Check if we have a Tokio runtime available
    if tokio::runtime::Handle::try_current().is_err() {
        return; // Still no runtime, skip for now
    }
    
    // Use INIT to ensure we only initialize once
    INIT.call_once(|| {
        if let Err(e) = setup_logging() {
            eprintln!("Failed to initialize OTEL logging: {}", e);
            // Don't exit here since we're in request processing
        } else {
            LAZY_OTEL_INITIALIZED.store(true, Ordering::Relaxed);
            tracing::info!("OTEL tracing initialized lazily");
        }
    });
}

/// Initialize the logger
pub fn init() {
    println!("init called");
    
    if jsonl_logging_enabled() {
        // For JSONL logging, defer initialization until Tokio runtime is available
        println!("JSONL logging enabled - deferring initialization until runtime is available");
        return;
    }
    
    // For non-JSONL logging, initialize immediately
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
// TODO: Add a subroutine to format and return the otel layer here
fn setup_logging() ->Result<(), Box<dyn std::error::Error>> {
    let fmt_filter_layer = filters(load_config());
    let trace_filter_layer = filters(load_config());
    if jsonl_logging_enabled() {
        // Setup tracer
        // TODO P2: Move this into its own function
        let otlp_exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint("http://localhost:4317")
            .with_timeout(std::time::Duration::from_secs(2)); // Add timeout
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(otlp_exporter)
            .with_trace_config(
                opentelemetry_sdk::trace::config()
                    .with_resource(opentelemetry_sdk::Resource::new(vec![
                        opentelemetry::KeyValue::new("service.name", "rust-tracing-playground"),
                        opentelemetry::KeyValue::new("service.version", "1.0.0"),
                    ]))
            )
            .install_batch(opentelemetry_sdk::runtime::Tokio)
            .map_err(|e| {
                eprintln!("Failed to initialize OTLP tracer: {}", e);
                e
            })?;

        let l = fmt::layer()
            .with_ansi(false)
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .event_format(CustomJsonFormatter::new())
            .with_writer(std::io::stderr)
            .with_filter(fmt_filter_layer);

        tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(tracer))
            .with(DistributedTraceIdLayer.with_filter(trace_filter_layer))
            .with(l)
            .init();
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
        .with_env_var(FILTER_ENV)
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
    let config_path = std::env::var(CONFIG_PATH_ENV).unwrap_or_else(|_| "".to_string());
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

    #[tracing::instrument(
        skip_all,
        fields(
            span_id = "abd16e319329445f",
            trace_id = "2adfd24468724599bb9a4990dc342288"
        )
    )]
    async fn parent() {
        tracing::Span::current().record("trace_id", "invalid");
        tracing::Span::current().record("span_id", "invalid");
        tracing::Span::current().record("span_name", "invalid");
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
            [("DYN_LOGGING_JSONL", Some("1"))],
            (async || {
                let tmp_file = NamedTempFile::new().unwrap();
                let file_name = tmp_file.path().to_str().unwrap();
                let guard = StderrOverride::from_file(file_name)?;
                init();
                parent().await;
                drop(guard);

                let lines = load_log(file_name)?;

                // 1. Validate my_trace_id matches parent's trace ID
                let parent_trace_id = Uuid::parse_str("2adfd24468724599bb9a4990dc342288")
                    .unwrap()
                    .simple()
                    .to_string();
                for log_line in &lines {
                    if let Some(my_trace_id) = log_line.get("my_trace_id") {
                        assert_eq!(
                            my_trace_id,
                            &serde_json::Value::String(parent_trace_id.clone())
                        );
                    }
                }

                // 2. Validate span IDs are unique for SPAN_CREATED and SPAN_CLOSED events
                let mut created_span_ids: Vec<String> = Vec::new();
                let mut closed_span_ids: Vec<String> = Vec::new();

                for log_line in &lines {
                    if let Some(message) = log_line.get("message") {
                        match message.as_str().unwrap() {
                            "SPAN_CREATED" => {
                                if let Some(span_id) = log_line.get("span_id") {
                                    let span_id_str = span_id.as_str().unwrap();
                                    assert!(
                                        created_span_ids.iter().all(|id| id != span_id_str),
                                        "Duplicate span ID found in SPAN_CREATED: {}",
                                        span_id_str
                                    );
                                    created_span_ids.push(span_id_str.to_string());
                                }
                            }
                            "SPAN_CLOSED" => {
                                if let Some(span_id) = log_line.get("span_id") {
                                    let span_id_str = span_id.as_str().unwrap();
                                    assert!(
                                        closed_span_ids.iter().all(|id| id != span_id_str),
                                        "Duplicate span ID found in SPAN_CLOSED: {}",
                                        span_id_str
                                    );
                                    closed_span_ids.push(span_id_str.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                }

                // Additionally, ensure that every SPAN_CLOSED has a corresponding SPAN_CREATED
                for closed_span_id in &closed_span_ids {
                    assert!(
                        created_span_ids.contains(closed_span_id),
                        "SPAN_CLOSED without corresponding SPAN_CREATED: {}",
                        closed_span_id
                    );
                }

                // 3. Validate parent span relationships
                let parent_span_id = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("message").unwrap().as_str().unwrap() == "SPAN_CREATED"
                            && log_line.get("span_name").unwrap().as_str().unwrap() == "parent"
                    })
                    .and_then(|log_line| {
                        log_line
                            .get("span_id")
                            .map(|s| s.as_str().unwrap().to_string())
                    })
                    .unwrap();

                let child_span_id = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("message").unwrap().as_str().unwrap() == "SPAN_CREATED"
                            && log_line.get("span_name").unwrap().as_str().unwrap() == "child"
                    })
                    .and_then(|log_line| {
                        log_line
                            .get("span_id")
                            .map(|s| s.as_str().unwrap().to_string())
                    })
                    .unwrap();

                let _grandchild_span_id = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("message").unwrap().as_str().unwrap() == "SPAN_CREATED"
                            && log_line.get("span_name").unwrap().as_str().unwrap() == "grandchild"
                    })
                    .and_then(|log_line| {
                        log_line
                            .get("span_id")
                            .map(|s| s.as_str().unwrap().to_string())
                    })
                    .unwrap();

                // Parent span has no parent_id
                for log_line in &lines {
                    if let Some(span_name) = log_line.get("span_name")
                        && let Some(span_name_str) = span_name.as_str()
                        && span_name_str == "parent"
                    {
                        assert!(log_line.get("parent_id").is_none());
                    }
                }

                // Child span's parent_id is parent_span_id
                for log_line in &lines {
                    if let Some(span_name) = log_line.get("span_name")
                        && let Some(span_name_str) = span_name.as_str()
                        && span_name_str == "child"
                    {
                        assert_eq!(
                            log_line.get("parent_id").unwrap().as_str().unwrap(),
                            &parent_span_id
                        );
                    }
                }

                // Grandchild span's parent_id is child_span_id
                for log_line in &lines {
                    if let Some(span_name) = log_line.get("span_name")
                        && let Some(span_name_str) = span_name.as_str()
                        && span_name_str == "grandchild"
                    {
                        assert_eq!(
                            log_line.get("parent_id").unwrap().as_str().unwrap(),
                            &child_span_id
                        );
                    }
                }

                // Validate duration relationships
                let parent_duration = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("message").unwrap().as_str().unwrap() == "SPAN_CLOSED"
                            && log_line.get("span_name").unwrap().as_str().unwrap() == "parent"
                    })
                    .and_then(|log_line| {
                        log_line
                            .get("time.duration_us")
                            .map(|d| d.as_u64().unwrap())
                    })
                    .unwrap();

                let child_duration = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("message").unwrap().as_str().unwrap() == "SPAN_CLOSED"
                            && log_line.get("span_name").unwrap().as_str().unwrap() == "child"
                    })
                    .and_then(|log_line| {
                        log_line
                            .get("time.duration_us")
                            .map(|d| d.as_u64().unwrap())
                    })
                    .unwrap();

                let grandchild_duration = lines
                    .iter()
                    .find(|log_line| {
                        log_line.get("message").unwrap().as_str().unwrap() == "SPAN_CLOSED"
                            && log_line.get("span_name").unwrap().as_str().unwrap() == "grandchild"
                    })
                    .and_then(|log_line| {
                        log_line
                            .get("time.duration_us")
                            .map(|d| d.as_u64().unwrap())
                    })
                    .unwrap();

                assert!(
                    parent_duration > child_duration + grandchild_duration,
                    "Parent duration is not greater than the sum of child and grandchild durations"
                );
                assert!(
                    child_duration > grandchild_duration,
                    "Child duration is not greater than grandchild duration"
                );

                Ok::<(), anyhow::Error>(())
            })(),
        )
        .await;
        Ok(())
    }

    // TODO: Add tests here
}

