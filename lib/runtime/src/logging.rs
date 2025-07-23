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
use tracing::Span;
use tracing_subscriber::layer::Context;
use tracing::Id;
use tracing::span;
use tracing_subscriber::Layer;
use tracing_subscriber::fmt::format::FmtSpan;
use uuid::Uuid;
use tracing::field::Field;
use tracing::span::Record;
use tracing_subscriber::field::Visit;
use tracing_subscriber::Registry;
use tracing_subscriber::registry::SpanData;
use crate::error;
use crate::config::{disable_ansi_logging, jsonl_logging_enabled};

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
/// A valid trace ID is a 16-character hexadecimal string (lowercase).
pub fn is_valid_trace_id(trace_id: &str) -> bool {
    trace_id.len() == 32 && trace_id.chars().all(|c| c.is_ascii_hexdigit())
}

/// Validate a given span ID according to W3C Trace Context specifications.
/// A valid span ID is an 8-character hexadecimal string (lowercase).
pub fn is_valid_span_id(span_id: &str) -> bool {
    span_id.len() == 16 && span_id.chars().all(|c| c.is_ascii_hexdigit())
}

pub struct DistributedTraceIdLayer;

#[derive(Clone)]
pub struct DistributedTracingContext {
    trace_id: String,
    span_id: String,
    parent_id: Option<String>
}


#[derive(Debug, Default)]
pub struct FieldVisitor {
    pub fields: HashMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
	self.fields.insert(field.name().to_string(),value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
	self.fields.insert(field.name().to_string(),format!("{:?}", value).to_string());
    }
}

impl<S> Layer<S> for DistributedTraceIdLayer
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {

        if let Some(span) = ctx.span(id) {

	    let mut trace_id:Option<String> = None;
	    let mut parent_id:Option<String> = None;
	    let mut span_id:Option<String> = None;
	    let target_fields = ["trace_id", "span_id", "parent_id"];
	    let mut visitor = FieldVisitor::default();
	    attrs.record(&mut visitor);

	    for field in attrs.fields().iter() {
		if target_fields.contains(&field.name()) {
		if !visitor.fields.contains_key(field.name()) {
		    tracing::error!("Field {} has no value any attempts to update will be ignored", field.name());

		}
		}
	    }

	    if let Some(trace_id_input) = visitor.fields.get("trace_id") {
		if !is_valid_trace_id(trace_id_input) {
		    tracing::error!("trace id  '{}' is not valid! Ignoring.", trace_id_input);
		} else {
		    trace_id = Some(trace_id_input.to_string());
		}
	    }

	    if let Some(span_id_input) = visitor.fields.get("span_id") {
		if !is_valid_span_id(span_id_input) {
		    tracing::error!("span id  '{}' is not valid! Ignoring.", span_id_input);
		} else {
		    span_id = Some(span_id_input.to_string());
		}
	    }

	    if let Some(parent_id_input) = visitor.fields.get("parent_id") {
		if !is_valid_span_id(parent_id_input) {
		    tracing::error!("parent id  '{}' is not valid! Ignoring.", parent_id_input);
		} else {
		    parent_id = Some(parent_id_input.to_string());
		}
	    }

	    if parent_id == None {
		if let Some(parent_span_id) = ctx.current_span().id() {
		    if let Some(parent_span) = ctx.span(parent_span_id) {
			// Access parent span data (e.g., name)
			let parent_ext = parent_span.extensions();
			if let Some(parent_tracing_context) = parent_ext.get::<DistributedTracingContext>() {
			    trace_id = Some(parent_tracing_context.trace_id.clone());
			    parent_id = Some(parent_tracing_context.span_id.clone());
			}

		    }
		}
	    }

	    if (parent_id != None || span_id != None) && trace_id == None {
		tracing::error!("parent id or span id are set but trace id is not set!")
	    }

	    if trace_id == None {
		trace_id = Some(generate_trace_id());
	    }

	    if span_id == None {
		span_id = Some(generate_span_id());
	    }

	    let mut extensions = span.extensions_mut();
	    extensions.insert(DistributedTracingContext {trace_id:trace_id.expect("Trace ID must be set"),
							 span_id:span_id.expect("Span ID must be set"),
							 parent_id:parent_id});
        }
    }
}

pub fn get_distributed_tracing_context() -> Option<DistributedTracingContext> {
    Span::current().with_subscriber(|(id, subscriber)| {
        subscriber
            .downcast_ref::<Registry>()
            .and_then(|registry| registry.span_data(&id))
            .and_then(|span_data| {
                let extensions = span_data.extensions();
                extensions.get::<DistributedTracingContext>().cloned()
            })
    }).flatten()
}

/// Initialize the logger

pub fn init() {
    INIT.call_once(setup_logging);
}

#[cfg(feature = "tokio-console")]
fn setup_logging() {
    // Start tokio-console server. Returns a tracing-subscriber Layer.
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
    // The generics mean we have to repeat everything. Each builder method returns a
    // specialized type.
    if jsonl_logging_enabled() {
        // JSON logger for NIM

        let l = fmt::layer()
            .with_ansi(false)
	    .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
            .event_format(CustomJsonFormatter::new())
            .with_writer(std::io::stderr)
            .with_filter(filter_layer);
        tracing_subscriber::registry().with(DistributedTraceIdLayer).with(l).init();
    } else {
        // Normal logging

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

    // apply the log_filters from the config files
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

// TODO: This should be merged into the global config (rust/common/src/config.rs) once we have it
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
    file_path: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line_number: Option<u32>,
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
                .format("%Y-%m-%dT%H:%M:%S%.3f%:z")
                .to_string()
        } else {
            chrono::Utc::now()
                .format("%Y-%m-%dT%H:%M:%S%.3fZ")
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

use regex::Regex;
use once_cell::sync::Lazy;
fn parse_tracing_duration(s: &str) -> Option<u64> {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r#"^["']?\s*([0-9.]+)\s*(µs|us|ns|ms|s)\s*["']?$"#
        ).unwrap()
    });
    let captures = RE.captures(s)?;
    let value: f64 = captures[1].parse().ok()?;
    let unit = &captures[2];
    match unit {
        "ns"         => Some((value / 1000.0) as u64),
        "µs" | "us"  => Some(value as u64),
        "ms"         => Some((value * 1000.0) as u64),
        "s"          => Some((value * 1_000_000.0) as u64),
        _            => None,
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
        event.record(&mut visitor);
        let message = visitor
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
		println!("name {}",name);
                visitor.fields.insert(
                    name.to_string(),
                    serde_json::Value::String(value.trim_matches('"').to_string()),
                );
            }


	    // Calculate combined duration
            let busy = visitor.fields.remove("time.busy").and_then(|v| {
		let busy_us = parse_tracing_duration(&v.to_string());
		println!("{:?} {:?}",v,busy_us);
		v.as_i64().map(|v| v as u128)
            });
            let idle = visitor.fields.remove("time.idle").and_then(|v| {
		println!("{:?}",v);
		v.as_i64().map(|v| v as u128)
            });

	    println!("{:?}",busy);

            visitor.fields.insert(
                "span_name".to_string(),
                serde_json::Value::String(span.name().to_string()),
            );

	    if let Some(tracing_context) = ext.get::<DistributedTracingContext>() {
		visitor.fields.insert("span_id".to_string(),serde_json::Value::String(tracing_context.span_id.clone()));
		visitor.fields.insert("trace_id".to_string(),serde_json::Value::String(tracing_context.trace_id.clone()));
		if let Some(parent_id) = tracing_context.parent_id.clone() {
		    visitor.fields.insert("parent_id".to_string(),serde_json::Value::String(parent_id));
		}
	    }
	    else {
		tracing::error!("Distributed Tracing Context not found, falling back to internal ids");
		visitor.fields.insert("span_id".to_string(),serde_json::Value::String(span.id().into_u64().to_string()));
		if let Some(parent) = span.parent() {
		    visitor.fields.insert("parent_id".to_string(),serde_json::Value::String(parent.id().into_u64().to_string()));
		}

	    }

            visitor.fields.insert(
                "span_name".to_string(),
                serde_json::Value::String(span.name().to_string()),
            );

        }

        let metadata = event.metadata();
        let log = JsonLog {
            level: metadata.level().to_string(),
            time: self.time_formatter.format_now(),
            file_path: if cfg!(debug_assertions) {
                metadata.file()
            } else {
                None
            },
            line_number: if cfg!(debug_assertions) {
                metadata.line()
            } else {
                None
            },
            message,
            fields: visitor.fields,
        };
        let json = serde_json::to_string(&log).unwrap();
        writeln!(writer, "{json}")
    }
}

// Visitor to collect fields
#[derive(Default)]
struct JsonVisitor {
    // BTreeMap so that it's sorted, and always prints in the same order
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
	println!("value {}",field.name());

        self.fields.insert(
            field.name().to_string(),
            serde_json::Value::String(value.to_string()),
        );
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
	println!("value {}",field.name());
        self.fields.insert(
            field.name().to_string(),
            // Infinite or NaN values are not JSON numbers, replace them with 0.
            // It's unlikely that we would log an inf or nan value.
            serde_json::Value::Number(Number::from_f64(value).unwrap_or(0.into())),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tracing::instrument(skip_all,fields(span_id="abd16e319329445f", trace_id="foo", request_id))]
    async fn foo_3() {

	println!("recording");
	tracing::Span::current().record("trace_id","goo");
	tracing::Span::current().record("span_id","goo");
	tracing::Span::current().record("span_name","olivia");


	if let Some (my_ctx) = get_distributed_tracing_context() {
	    println!("my context {}", my_ctx.trace_id);
	}

	tracing::trace!(
	    message="received two parts",
	    header=5,
	    data="foo"
        );

	foo_2().await;
    }

    #[tracing::instrument(skip_all)]
    async fn foo() {
	tracing::trace!(
	    message="received two parts",
	    header=5,
	    data="foo"
        );

	foo_2().await;
    }

    #[tracing::instrument(skip_all)]
    async fn foo_2() {
	tracing::trace!(
	    message="received two parts",
	    header=5,
	    data="foo"
        );

    }

    #[tokio::test]
    #[tracing::instrument(skip_all)]
    async fn test_span() {
	init();
//	foo().await;
	foo_3().await;
    }
}
