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
    let f = filters(load_config());
    // The generics mean we have to repeat everything. Each builder method returns a
    // specialized type.
    if jsonl_logging_enabled() {
        // JSON logger for NIM

        let l = fmt::layer()
            .with_ansi(false)
            .event_format(CustomJsonFormatter::new())
            .with_writer(std::io::stderr)
            .with_filter(f);
        tracing_subscriber::registry().with(l).init();
    } else {
        // Normal logging

        let l = fmt::layer()
            .with_ansi(!disable_ansi_logging())
            .event_format(fmt::format().compact().with_timer(TimeFormatter::new()))
            .with_writer(std::io::stderr)
            .with_filter(f);
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
                visitor.fields.insert(
                    name.to_string(),
                    serde_json::Value::String(value.trim_matches('"').to_string()),
                );
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
        self.fields.insert(
            field.name().to_string(),
            // Infinite or NaN values are not JSON numbers, replace them with 0.
            // It's unlikely that we would log an inf or nan value.
            serde_json::Value::Number(Number::from_f64(value).unwrap_or(0.into())),
        );
    }
}
