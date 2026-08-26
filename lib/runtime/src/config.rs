// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use derive_builder::Builder;
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;
use validator::Validate;

pub mod environment_names;

/// Default system host for health and metrics endpoints
const DEFAULT_SYSTEM_HOST: &str = "0.0.0.0";

/// Default system port for health and metrics endpoints (-1 = disabled)
const DEFAULT_SYSTEM_PORT: i16 = -1;

/// Default health endpoint paths
const DEFAULT_SYSTEM_HEALTH_PATH: &str = "/health";
const DEFAULT_SYSTEM_LIVE_PATH: &str = "/live";

/// Default health check configuration
/// This is the wait time before sending canary health checks when no activity is detected
pub const DEFAULT_CANARY_WAIT_TIME_SECS: u64 = 10;
/// Default timeout for individual health check requests
pub const DEFAULT_HEALTH_CHECK_REQUEST_TIMEOUT_SECS: u64 = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Grace shutdown period for the system server.
    pub graceful_shutdown_timeout: u64,
}

impl WorkerConfig {
    /// Instantiates and reads server configurations from appropriate sources.
    /// Panics on invalid configuration.
    pub fn from_settings() -> Self {
        // All calls should be global and thread safe.
        Figment::new()
            .merge(Serialized::defaults(Self::default()))
            .merge(Env::prefixed("DYN_WORKER_"))
            .extract()
            .unwrap() // safety: Called on startup, so panic is reasonable
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        WorkerConfig {
            graceful_shutdown_timeout: if cfg!(debug_assertions) {
                1 // Debug build: 1 second
            } else {
                30 // Release build: 30 seconds
            },
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Ready,
    NotReady,
}

/// Runtime configuration
/// Defines the configuration for Tokio runtimes
#[derive(Serialize, Deserialize, Validate, Debug, Builder, Clone)]
#[builder(build_fn(private, name = "build_internal"), derive(Debug, Serialize))]
pub struct RuntimeConfig {
    /// Number of async worker threads
    /// If set to 1, the runtime will run in single-threaded mode
    /// Set this at runtime with environment variable DYN_RUNTIME_NUM_WORKER_THREADS. Defaults to
    /// number of cores.
    #[validate(range(min = 1))]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub num_worker_threads: Option<usize>,

    /// Maximum number of blocking threads
    /// Blocking threads are used for blocking operations, this value must be greater than 0.
    /// Set this at runtime with environment variable DYN_RUNTIME_MAX_BLOCKING_THREADS.
    ///
    /// Defaults to the machine's core count -- see `impl Default for RuntimeConfig`, which sets
    /// `max_blocking_threads: num_cores`. The `#[builder(default = "512")]` below applies only when
    /// constructing through `RuntimeConfigBuilder` without setting this field; it is NOT the
    /// default a normally-configured process sees.
    ///
    /// This comment previously read "Defaults to 512", which is a good way to build an incorrect
    /// model of how many threads a process will start.
    ///
    /// Note this is a ceiling, not a preallocation: Tokio spawns blocking threads on demand and
    /// reaps them after an idle timeout, so an observed thread count is timing-sensitive and should
    /// be taken at steady state under load.
    #[validate(range(min = 1))]
    #[builder(default = "512")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub max_blocking_threads: usize,

    /// System status server host for health and metrics endpoints
    /// Set this at runtime with environment variable DYN_SYSTEM_HOST
    #[builder(default = "DEFAULT_SYSTEM_HOST.to_string()")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub system_host: String,

    /// System status server port for health and metrics endpoints
    /// Set to -1 to disable the system status server (default)
    /// Set to 0 to bind to a random available port
    /// Set to a positive port number (e.g. 8081) to bind to a specific port
    /// Set this at runtime with environment variable DYN_SYSTEM_PORT
    #[builder(default = "DEFAULT_SYSTEM_PORT")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub system_port: i16,

    /// Health and metrics System status server enabled (DEPRECATED)
    /// This field is deprecated. Use system_port instead (set to positive value to enable)
    /// Environment variable DYN_SYSTEM_ENABLED is deprecated
    #[deprecated(
        note = "Use system_port instead. Set DYN_SYSTEM_PORT to enable the system metrics server."
    )]
    #[builder(default = "false")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub system_enabled: bool,

    /// Starting Health Status
    /// Set this at runtime with environment variable DYN_SYSTEM_STARTING_HEALTH_STATUS
    #[builder(default = "HealthStatus::NotReady")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub starting_health_status: HealthStatus,

    /// Use Endpoint Health Status
    /// When using endpoint health status, health status
    /// is the AND of individual endpoint health
    /// Set this at runtime with environment variable DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS
    /// with the list of endpoints to consider for system health
    #[builder(default = "vec![]")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub use_endpoint_health_status: Vec<String>,

    /// Health endpoint paths
    /// Set this at runtime with environment variable DYN_SYSTEM_HEALTH_PATH
    #[builder(default = "DEFAULT_SYSTEM_HEALTH_PATH.to_string()")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub system_health_path: String,
    /// Set this at runtime with environment variable DYN_SYSTEM_LIVE_PATH
    #[builder(default = "DEFAULT_SYSTEM_LIVE_PATH.to_string()")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub system_live_path: String,

    /// Number of threads for the Rayon compute pool
    /// If not set, defaults to num_cpus / 2
    /// Set this at runtime with environment variable DYN_COMPUTE_THREADS
    #[builder(default = "None")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub compute_threads: Option<usize>,

    /// Stack size for compute threads in bytes
    /// Defaults to 2MB (2097152 bytes)
    /// Set this at runtime with environment variable DYN_COMPUTE_STACK_SIZE
    #[builder(default = "Some(2 * 1024 * 1024)")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub compute_stack_size: Option<usize>,

    /// Thread name prefix for compute pool threads
    /// Set this at runtime with environment variable DYN_COMPUTE_THREAD_PREFIX
    #[builder(default = "\"compute\".to_string()")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub compute_thread_prefix: String,

    /// Enable active health checking with payloads
    /// Set this at runtime with environment variable DYN_HEALTH_CHECK_ENABLED
    #[builder(default = "false")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub health_check_enabled: bool,

    /// Canary wait time in seconds (time to wait before sending health check when no activity)
    /// Set this at runtime with environment variable DYN_CANARY_WAIT_TIME
    #[builder(default = "DEFAULT_CANARY_WAIT_TIME_SECS")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub canary_wait_time_secs: u64,

    /// Health check request timeout in seconds
    /// Set this at runtime with environment variable DYN_HEALTH_CHECK_REQUEST_TIMEOUT
    #[builder(default = "DEFAULT_HEALTH_CHECK_REQUEST_TIMEOUT_SECS")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub health_check_request_timeout_secs: u64,
}

impl fmt::Display for RuntimeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If None, it defaults to "number of cores", so we indicate that.
        match self.num_worker_threads {
            Some(val) => write!(f, "num_worker_threads={val}, ")?,
            None => write!(f, "num_worker_threads=default (num_cores), ")?,
        }

        write!(f, "max_blocking_threads={}, ", self.max_blocking_threads)?;
        write!(f, "system_host={}, ", self.system_host)?;
        write!(f, "system_port={}, ", self.system_port)?;
        write!(
            f,
            "use_endpoint_health_status={:?}",
            self.use_endpoint_health_status
        )?;
        write!(
            f,
            "starting_health_status={:?}",
            self.starting_health_status
        )?;
        write!(f, ", system_health_path={}", self.system_health_path)?;
        write!(f, ", system_live_path={}", self.system_live_path)?;
        write!(f, ", health_check_enabled={}", self.health_check_enabled)?;
        write!(f, ", canary_wait_time_secs={}", self.canary_wait_time_secs)?;
        write!(
            f,
            ", health_check_request_timeout_secs={}",
            self.health_check_request_timeout_secs
        )?;

        Ok(())
    }
}

impl RuntimeConfig {
    pub fn builder() -> RuntimeConfigBuilder {
        RuntimeConfigBuilder::default()
    }

    pub(crate) fn figment() -> Figment {
        Figment::new()
            .merge(Serialized::defaults(RuntimeConfig::default()))
            .merge(Toml::file("/opt/dynamo/defaults/runtime.toml"))
            .merge(Toml::file("/opt/dynamo/etc/runtime.toml"))
            .merge(Env::prefixed("DYN_RUNTIME_").filter_map(|k| {
                let full_key = format!("DYN_RUNTIME_{}", k.as_str());
                // filters out empty environment variables
                match std::env::var(&full_key) {
                    Ok(v) if !v.is_empty() => Some(k.into()),
                    _ => None,
                }
            }))
            .merge(Env::prefixed("DYN_SYSTEM_").filter_map(|k| {
                let full_key = format!("DYN_SYSTEM_{}", k.as_str());
                // filters out empty environment variables
                match std::env::var(&full_key) {
                    Ok(v) if !v.is_empty() => {
                        // Map DYN_SYSTEM_* to the correct field names
                        let mapped_key = match k.as_str() {
                            "HOST" => "system_host",
                            "PORT" => "system_port",
                            "ENABLED" => "system_enabled",
                            "USE_ENDPOINT_HEALTH_STATUS" => "use_endpoint_health_status",
                            "STARTING_HEALTH_STATUS" => "starting_health_status",
                            "HEALTH_PATH" => "system_health_path",
                            "LIVE_PATH" => "system_live_path",
                            _ => k.as_str(),
                        };
                        Some(mapped_key.into())
                    }
                    _ => None,
                }
            }))
            .merge(Env::prefixed("DYN_COMPUTE_").filter_map(|k| {
                let full_key = format!("DYN_COMPUTE_{}", k.as_str());
                // filters out empty environment variables
                match std::env::var(&full_key) {
                    Ok(v) if !v.is_empty() => {
                        // Map DYN_COMPUTE_* to the correct field names
                        let mapped_key = match k.as_str() {
                            "THREADS" => "compute_threads",
                            "STACK_SIZE" => "compute_stack_size",
                            "THREAD_PREFIX" => "compute_thread_prefix",
                            _ => k.as_str(),
                        };
                        Some(mapped_key.into())
                    }
                    _ => None,
                }
            }))
            .merge(Env::prefixed("DYN_HEALTH_CHECK_").filter_map(|k| {
                let full_key = format!("DYN_HEALTH_CHECK_{}", k.as_str());
                // filters out empty environment variables
                match std::env::var(&full_key) {
                    Ok(v) if !v.is_empty() => {
                        // Map DYN_HEALTH_CHECK_* to the correct field names
                        let mapped_key = match k.as_str() {
                            "ENABLED" => "health_check_enabled",
                            "REQUEST_TIMEOUT" => "health_check_request_timeout_secs",
                            _ => k.as_str(),
                        };
                        Some(mapped_key.into())
                    }
                    _ => None,
                }
            }))
            .merge(Env::prefixed("DYN_CANARY_").filter_map(|k| {
                let full_key = format!("DYN_CANARY_{}", k.as_str());
                // filters out empty environment variables
                match std::env::var(&full_key) {
                    Ok(v) if !v.is_empty() => {
                        // Map DYN_CANARY_* to the correct field names
                        let mapped_key = match k.as_str() {
                            "WAIT_TIME" => "canary_wait_time_secs",
                            _ => k.as_str(),
                        };
                        Some(mapped_key.into())
                    }
                    _ => None,
                }
            }))
    }

    /// Load the runtime configuration from the environment and configuration files
    /// Configuration is priorities in the following order, where the last has the lowest priority:
    /// 1. Environment variables (top priority)
    ///    TO DO: Add documentation for configuration files. Paths should be configurable.
    /// 2. /opt/dynamo/etc/runtime.toml
    /// 3. /opt/dynamo/defaults/runtime.toml (lowest priority)
    ///
    /// Environment variables are prefixed with `DYN_RUNTIME_` and `DYN_SYSTEM`
    pub fn from_settings() -> Result<RuntimeConfig> {
        use environment_names::runtime::system as env_system;
        // Check for deprecated environment variables
        if std::env::var(env_system::DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS).is_ok() {
            tracing::warn!(
                "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS is deprecated and no longer used. \
                System health is now determined by endpoints that register with health check payloads. \
                Please update your configuration to register health check payloads directly on endpoints."
            );
        }

        if std::env::var(env_system::DYN_SYSTEM_ENABLED).is_ok() {
            tracing::warn!(
                "DYN_SYSTEM_ENABLED is deprecated. \
                System metrics server is now controlled solely by DYN_SYSTEM_PORT. \
                Set DYN_SYSTEM_PORT to a positive value to enable the server, or set to -1 to disable (default)."
            );
        }

        let config: RuntimeConfig = Self::figment().extract()?;
        config.validate()?;
        Ok(config)
    }

    /// Check if System server should be enabled
    /// System server is enabled when DYN_SYSTEM_PORT is set to 0 or a positive value
    /// Port 0 binds to a random available port
    /// Negative values disable the server
    pub fn system_server_enabled(&self) -> bool {
        self.system_port >= 0
    }

    pub fn single_threaded() -> Self {
        RuntimeConfig {
            num_worker_threads: Some(1),
            max_blocking_threads: 1,
            system_host: DEFAULT_SYSTEM_HOST.to_string(),
            system_port: DEFAULT_SYSTEM_PORT,
            #[allow(deprecated)]
            system_enabled: false,
            starting_health_status: HealthStatus::NotReady,
            use_endpoint_health_status: vec![],
            system_health_path: DEFAULT_SYSTEM_HEALTH_PATH.to_string(),
            system_live_path: DEFAULT_SYSTEM_LIVE_PATH.to_string(),
            compute_threads: Some(1),
            compute_stack_size: Some(2 * 1024 * 1024),
            compute_thread_prefix: "compute".to_string(),
            health_check_enabled: false,
            canary_wait_time_secs: DEFAULT_CANARY_WAIT_TIME_SECS,
            health_check_request_timeout_secs: DEFAULT_HEALTH_CHECK_REQUEST_TIMEOUT_SECS,
        }
    }

    /// Create a new default runtime configuration
    pub(crate) fn create_runtime(&self) -> std::io::Result<tokio::runtime::Runtime> {
        let mut builder = tokio::runtime::Builder::new_multi_thread();
        builder
            .worker_threads(
                self.num_worker_threads
                    .unwrap_or_else(|| std::thread::available_parallelism().unwrap().get()),
            )
            .max_blocking_threads(self.max_blocking_threads)
            .enable_all();
        if env_is_truthy(environment_names::runtime::DYN_ENABLE_POLL_HISTOGRAM) {
            tracing::info!(
                "Tokio poll-time histogram enabled (DYN_ENABLE_POLL_HISTOGRAM); \
                 expect ~2× Instant::now() overhead per task poll"
            );
            builder.enable_metrics_poll_time_histogram();
        }
        builder.build()
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let num_cores = std::thread::available_parallelism().unwrap().get();
        Self {
            num_worker_threads: Some(num_cores),
            max_blocking_threads: num_cores,
            system_host: DEFAULT_SYSTEM_HOST.to_string(),
            system_port: DEFAULT_SYSTEM_PORT,
            #[allow(deprecated)]
            system_enabled: false,
            starting_health_status: HealthStatus::NotReady,
            use_endpoint_health_status: vec![],
            system_health_path: DEFAULT_SYSTEM_HEALTH_PATH.to_string(),
            system_live_path: DEFAULT_SYSTEM_LIVE_PATH.to_string(),
            compute_threads: None,
            compute_stack_size: Some(2 * 1024 * 1024),
            compute_thread_prefix: "compute".to_string(),
            health_check_enabled: false,
            canary_wait_time_secs: DEFAULT_CANARY_WAIT_TIME_SECS,
            health_check_request_timeout_secs: DEFAULT_HEALTH_CHECK_REQUEST_TIMEOUT_SECS,
        }
    }
}

impl RuntimeConfigBuilder {
    /// Build and validate the runtime configuration
    pub fn build(&self) -> Result<RuntimeConfig> {
        let config = self.build_internal()?;
        config.validate()?;
        Ok(config)
    }
}

// Canonical truthy/falsy/bool parsing for user-supplied configuration
// (environment variables, headers, config values). The single implementation
// lives in the zero-dependency `dynamo-truthy` crate so that crates which
// cannot depend on `dynamo-runtime` share it too; this re-export is the
// canonical import path for everything that can.
pub use dynamo_truthy::{
    env_is_falsey, env_is_truthy, is_falsey, is_truthy, parse_bool, parse_bool_opt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleLogFormat {
    Readable,
    Jsonl,
}

impl ConsoleLogFormat {
    fn from_env_value(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "readable" => Some(Self::Readable),
            "jsonl" => Some(Self::Jsonl),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Readable => "readable",
            Self::Jsonl => "jsonl",
        }
    }
}

/// Return whether the legacy `DYN_LOGGING_JSONL` switch is enabled.
///
/// This remains a separate compatibility signal because older deployments
/// also use it to enable local trace-context propagation.
pub(crate) fn legacy_jsonl_logging_enabled() -> bool {
    env_is_truthy(environment_names::logging::DYN_LOGGING_JSONL)
}

/// Return the console log format.
///
/// `DYN_LOGGING_CONSOLE_FORMAT` takes precedence. `DYN_LOGGING_JSONL` remains
/// supported as a legacy fallback when the new setting is unset or blank.
pub fn console_log_format() -> ConsoleLogFormat {
    let legacy_format = || {
        if legacy_jsonl_logging_enabled() {
            ConsoleLogFormat::Jsonl
        } else {
            ConsoleLogFormat::Readable
        }
    };

    match std::env::var(environment_names::logging::DYN_LOGGING_CONSOLE_FORMAT) {
        Ok(value) if value.trim().is_empty() => legacy_format(),
        Ok(value) => match ConsoleLogFormat::from_env_value(value.trim()) {
            Some(format) => format,
            None => {
                eprintln!(
                    "Invalid {} value '{}'; using readable console logs",
                    environment_names::logging::DYN_LOGGING_CONSOLE_FORMAT,
                    value
                );
                ConsoleLogFormat::Readable
            }
        },
        Err(_) => legacy_format(),
    }
}

/// Return whether the effective console log format is JSONL.
pub fn jsonl_logging_enabled() -> bool {
    console_log_format() == ConsoleLogFormat::Jsonl
}

/// Check whether logging with ANSI terminal escape codes and colors is disabled.
/// Set the `DYN_SDK_DISABLE_ANSI_LOGGING` environment variable a [`is_truthy`] value
pub fn disable_ansi_logging() -> bool {
    env_is_truthy(environment_names::logging::DYN_SDK_DISABLE_ANSI_LOGGING)
}

/// Check whether to use local timezone for logging timestamps (default is UTC)
/// Set the `DYN_LOG_USE_LOCAL_TZ` environment variable to a [`is_truthy`] value
pub fn use_local_timezone() -> bool {
    env_is_truthy(environment_names::logging::DYN_LOG_USE_LOCAL_TZ)
}

/// Returns true if `DYN_LOGGING_SPAN_EVENTS` is set to a truthy value.
pub fn span_events_enabled() -> bool {
    env_is_truthy(environment_names::logging::DYN_LOGGING_SPAN_EVENTS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_builder_overrides_related_fields() -> Result<()> {
        let config = RuntimeConfig::builder()
            .num_worker_threads(Some(24))
            .max_blocking_threads(32)
            .system_host("127.0.0.1".to_string())
            .system_port(9090)
            .build()?;

        assert_eq!(config.num_worker_threads, Some(24));
        assert_eq!(config.max_blocking_threads, 32);
        assert_eq!(config.system_host, "127.0.0.1");
        assert_eq!(config.system_port, 9090);
        Ok(())
    }

    /// Both thread-pool variables must survive `from_settings()`.
    ///
    /// Regression guard. It has previously been unclear whether
    /// `DYN_RUNTIME_MAX_BLOCKING_THREADS` was read at all, because setting it had
    /// no observable effect on a frontend's thread count. The parsing was in fact
    /// correct -- the runtime it configured was not the one serving traffic -- but
    /// nothing pinned down the parsing side, so the question stayed open longer
    /// than it needed to.
    ///
    /// `#[serial]` is not available here, so this test sets and restores the
    /// variables itself and must not be run concurrently with another test that
    /// touches the same ones. It is currently the only test that does.
    #[test]
    fn test_from_settings_reads_both_thread_env_vars() -> Result<()> {
        const WORKERS: &str = "DYN_RUNTIME_NUM_WORKER_THREADS";
        const BLOCKING: &str = "DYN_RUNTIME_MAX_BLOCKING_THREADS";

        let prev_workers = std::env::var(WORKERS).ok();
        let prev_blocking = std::env::var(BLOCKING).ok();

        // SAFETY: single-threaded within this test; see the note above about
        // concurrent tests touching these variables.
        unsafe {
            std::env::set_var(WORKERS, "7");
            std::env::set_var(BLOCKING, "11");
        }

        let config = RuntimeConfig::from_settings();

        unsafe {
            match prev_workers {
                Some(v) => std::env::set_var(WORKERS, v),
                None => std::env::remove_var(WORKERS),
            }
            match prev_blocking {
                Some(v) => std::env::set_var(BLOCKING, v),
                None => std::env::remove_var(BLOCKING),
            }
        }

        let config = config?;
        assert_eq!(config.num_worker_threads, Some(7), "{WORKERS} was not read");
        assert_eq!(config.max_blocking_threads, 11, "{BLOCKING} was not read");
        Ok(())
    }

    /// `Default` sets `max_blocking_threads` to the core count, NOT 512.
    ///
    /// The doc comment on the field used to say 512 -- that is the
    /// `#[builder(default)]`, which applies only when building through
    /// `RuntimeConfigBuilder`. Reading the comment instead of the impl is an easy way to
    /// mispredict how many threads a process will start.
    #[test]
    fn test_default_max_blocking_threads_is_core_count() {
        let cores = std::thread::available_parallelism().unwrap().get();
        let config = RuntimeConfig::default();
        assert_eq!(config.max_blocking_threads, cores);
        assert_eq!(config.num_worker_threads, Some(cores));
    }

    #[test]
    fn test_runtime_config_rejects_invalid_thread_count() -> Result<()> {
        let result = RuntimeConfig::builder()
            .num_worker_threads(Some(0))
            .max_blocking_threads(0)
            .build();

        let error = result.unwrap_err().to_string();
        assert!(error.contains("num_worker_threads: Validation error"));
        assert!(error.contains("max_blocking_threads: Validation error"));
        Ok(())
    }

    #[test]
    fn test_system_server_enabled_by_nonnegative_port() {
        let mut config = RuntimeConfig::default();
        for (port, enabled) in [(-1, false), (0, true), (9527, true)] {
            config.system_port = port;
            assert_eq!(config.system_server_enabled(), enabled);
        }
    }

    #[test]
    fn test_is_truthy_and_falsey() {
        // Test truthy values
        assert!(is_truthy("1"));
        assert!(is_truthy("true"));
        assert!(is_truthy("TRUE"));
        assert!(is_truthy("on"));
        assert!(is_truthy("yes"));

        // Test falsey values
        assert!(is_falsey("0"));
        assert!(is_falsey("false"));
        assert!(is_falsey("FALSE"));
        assert!(is_falsey("off"));
        assert!(is_falsey("no"));

        // Test opposite behavior
        assert!(!is_truthy("0"));
        assert!(!is_falsey("1"));
    }

    #[test]
    fn test_console_log_format() {
        use environment_names::logging;

        for (console_format, legacy_jsonl, expected) in [
            (None, None, ConsoleLogFormat::Readable),
            (None, Some("true"), ConsoleLogFormat::Jsonl),
            (Some(""), Some("true"), ConsoleLogFormat::Jsonl),
            (Some("   "), Some("true"), ConsoleLogFormat::Jsonl),
            (Some(" jsonl "), Some("false"), ConsoleLogFormat::Jsonl),
            (Some("readable"), Some("true"), ConsoleLogFormat::Readable),
            (Some("jsonl"), Some("false"), ConsoleLogFormat::Jsonl),
            (
                Some("unsupported"),
                Some("true"),
                ConsoleLogFormat::Readable,
            ),
        ] {
            temp_env::with_vars(
                [
                    (logging::DYN_LOGGING_CONSOLE_FORMAT, console_format),
                    (logging::DYN_LOGGING_JSONL, legacy_jsonl),
                ],
                || {
                    assert_eq!(console_log_format(), expected);
                    assert_eq!(jsonl_logging_enabled(), expected == ConsoleLogFormat::Jsonl);
                },
            );
        }
    }
}
