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

use figment::{
    providers::{Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use tracing::level_filters::LevelFilter;
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::time::FormatTime;
use tracing_subscriber::fmt::time::LocalTime;
use tracing_subscriber::fmt::time::SystemTime;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::fmt::{format::Writer, FormattedFields};
use tracing_subscriber::fmt::{FmtContext, FormatFields};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::{filter::Directive, fmt};

use crate::config::{disable_ansi_logging, jsonl_logging_enabled};
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use serde_json::Value;
use std::convert::Infallible;
use std::time::Instant;
use tracing::field::Field;
use tracing::span;
use tracing::Id;
use tracing::Span;
use tracing_subscriber::field::Visit;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::SpanData;
use tracing_subscriber::Layer;
use tracing_subscriber::Registry;
use uuid::Uuid;

/// ENV used to set the log level
const FILTER_ENV: &str = "DYN_LOG";

/// Default log level
const DEFAULT_FILTER_LEVEL: &str = "info";

/// ENV used to set the path to the logging configuration file
const CONFIG_PATH_ENV: &str = "DYN_LOGGING_CONFIG_PATH";

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

#[derive(Clone)]
pub struct DistributedTraceContext {
    trace_id: String,
    span_id: String,
    parent_id: Option<String>,
    tracestate: Option<String>,
    start: Instant,
    end: Option<Instant>,
    x_request_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TraceParent {
    pub trace_id: Option<String>,
    pub parent_id: Option<String>,
    pub tracestate: Option<String>,
    pub x_request_id: Option<String>,
}

impl<S> FromRequestParts<S> for TraceParent
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let mut trace_id = None;
        let mut parent_id = None;
        let mut tracestate = None;
        if let Some(header_value) = parts.headers.get("traceparent") {
            if let Ok(header_str) = header_value.to_str() {
                let pieces: Vec<_> = header_str.split('-').collect();
                if pieces.len() == 4 {
                    let candidate_trace_id = pieces[1];
                    let candidate_parent_id = pieces[2];

                    if is_valid_trace_id(candidate_trace_id)
                        && is_valid_span_id(candidate_parent_id)
                    {
                        trace_id = Some(candidate_trace_id.to_string());
                        parent_id = Some(candidate_parent_id.to_string());
                    } else {
                        tracing::debug!("Invalid traceparent header: {}", header_str);
                    }
                }
            }
        }

        if let Some(header_value) = parts.headers.get("tracestate") {
            if let Ok(header_str) = header_value.to_str() {
                tracestate = Some(header_str.to_string());
            }
        }

        // Extract X-Request-ID or x-request-id (case-insensitive)
        let x_request_id = parts
            .headers
            .get("x-request-id")
            .and_then(|val| val.to_str().ok())
            .map(|s| s.to_string());

        Ok(TraceParent {
            trace_id,
            parent_id,
            tracestate,
            x_request_id,
        })
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

    // Adds W3C compliant span_id, trace_id, and parent_id if not already present
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut trace_id: Option<String> = None;
            let mut parent_id: Option<String> = None;
            let mut span_id: Option<String> = None;
            let mut x_request_id: Option<String> = None;
            let mut tracestate: Option<String> = None;
            let mut visitor = FieldVisitor::default();
            attrs.record(&mut visitor);

            if let Some(trace_id_input) = visitor.fields.get("trace_id") {
                if !is_valid_trace_id(trace_id_input) {
                    tracing::trace!("trace id  '{}' is not valid! Ignoring.", trace_id_input);
                } else {
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

            if parent_id.is_none() {
                if let Some(parent_span_id) = ctx.current_span().id() {
                    if let Some(parent_span) = ctx.span(parent_span_id) {
                        let parent_ext = parent_span.extensions();
                        if let Some(parent_tracing_context) =
                            parent_ext.get::<DistributedTraceContext>()
                        {
                            trace_id = Some(parent_tracing_context.trace_id.clone());
                            parent_id = Some(parent_tracing_context.span_id.clone());
                            tracestate = parent_tracing_context.tracestate.clone();
                        }
                    }
                }
            }

            if (parent_id.is_some() || span_id.is_some()) && trace_id.is_none() {
                tracing::error!("parent id or span id are set but trace id is not set!");
                // Clear inconsistent IDs to maintain trace integrity
                parent_id = None;
                span_id = None;
            }

            if trace_id.is_none() {
                trace_id = Some(generate_trace_id());
            }
            if span_id.is_none() {
                span_id = Some(generate_span_id());
            }

            let mut extensions = span.extensions_mut();
            extensions.insert(DistributedTraceContext {
                trace_id: trace_id.expect("Trace ID must be set"),
                span_id: span_id.expect("Span ID must be set"),
                parent_id,
                tracestate,
                start: Instant::now(),
                end: None,
                x_request_id,
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

/// Initialize the logger
pub fn init() {
    INIT.call_once(setup_logging);
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
fn setup_logging() {
    let filter_layer = filters(load_config());
    if jsonl_logging_enabled() {
        let l = fmt::layer()
            .with_ansi(false)
            .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .event_format(CustomJsonFormatter::new())
            .with_writer(std::io::stderr)
            .with_filter(filter_layer);
        tracing_subscriber::registry()
            .with(DistributedTraceIdLayer)
            .with(l)
            .init();
    } else {
        let l = fmt::layer()
            .with_ansi(!disable_ansi_logging())
            .event_format(fmt::format().compact().with_timer(TimeFormatter::new()))
            .with_writer(std::io::stderr)
            .with_filter(filter_layer);
        tracing_subscriber::registry().with(l).init();
    }
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
    use anyhow::{anyhow, Result};
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
                    if log_line.get("span_name").unwrap().as_str().unwrap() == "parent" {
                        assert!(log_line.get("parent_id").is_none());
                    }
                }

                // Child span's parent_id is parent_span_id
                for log_line in &lines {
                    if log_line.get("span_name").unwrap().as_str().unwrap() == "child" {
                        assert_eq!(
                            log_line.get("parent_id").unwrap().as_str().unwrap(),
                            &parent_span_id
                        );
                    }
                }

                // Grandchild span's parent_id is child_span_id
                for log_line in &lines {
                    if log_line.get("span_name").unwrap().as_str().unwrap() == "grandchild" {
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
}
