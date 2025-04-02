// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::Result;
use derive_builder::Builder;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use serde::{Deserialize, Serialize};
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Grace shutdown period for http-service.
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

/// Runtime configuration
/// Defines the configuration for Tokio runtimes
#[derive(Serialize, Deserialize, Validate, Debug, Builder, Clone)]
#[builder(build_fn(private, name = "build_internal"), derive(Debug, Serialize))]
pub struct RuntimeConfig {
    /// Number of async worker threads
    /// If set to 1, the runtime will run in single-threaded mode
    #[validate(range(min = 1))]
    #[builder(default = "16")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub num_worker_threads: usize,

    /// Maximum number of blocking threads
    /// Blocking threads are used for blocking operations, this value must be greater than 0.
    #[validate(range(min = 1))]
    #[builder(default = "512")]
    #[builder_field_attr(serde(skip_serializing_if = "Option::is_none"))]
    pub max_blocking_threads: usize,
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
    }

    /// Load the runtime configuration from the environment and configuration files
    /// Configuration is priorities in the following order, where the last has the lowest priority:
    /// 1. Environment variables (top priority)
    ///     TO DO: Add documentation for configuration files. Paths should be configurable.
    /// 2. /opt/dynamo/etc/runtime.toml
    /// 3. /opt/dynamo/defaults/runtime.toml (lowest priority)
    ///
    /// Environment variables are prefixed with `DYN_RUNTIME_`
    pub fn from_settings() -> Result<RuntimeConfig> {
        let config: RuntimeConfig = Self::figment().extract()?;
        config.validate()?;
        Ok(config)
    }

    pub fn single_threaded() -> Self {
        RuntimeConfig {
            num_worker_threads: 1,
            max_blocking_threads: 1,
        }
    }

    /// Create a new default runtime configuration
    pub(crate) fn create_runtime(&self) -> Result<tokio::runtime::Runtime> {
        Ok(tokio::runtime::Builder::new_multi_thread()
            .worker_threads(self.num_worker_threads)
            .max_blocking_threads(self.max_blocking_threads)
            .enable_all()
            .build()?)
    }
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_worker_threads: 16,
            max_blocking_threads: 16,
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

/// Check if an environment variable is truthy
pub fn env_is_truthy(env: &str) -> bool {
    match std::env::var(env) {
        Ok(val) => is_truthy(val.as_str()),
        Err(_) => false,
    }
}

/// Check if a string is truthy
/// This will be used to evaluate environment variables or any other subjective
/// configuration parameters that can be set by the user that should be evaluated
/// as a boolean value.
pub fn is_truthy(val: &str) -> bool {
    matches!(val.to_lowercase().as_str(), "1" | "true" | "on" | "yes")
}

/// Check whether JSONL logging enabled
/// Set the `DYN_LOGGING_JSONL` environment variable a [`is_truthy`] value
pub fn jsonl_logging_enabled() -> bool {
    env_is_truthy("DYN_LOGGING_JSONL")
}

/// Check whether logging with ANSI terminal escape codes and colors is disabled.
/// Set the `DYN_SDK_DISABLE_ANSI_LOGGING` environment variable a [`is_truthy`] value
pub fn disable_ansi_logging() -> bool {
    env_is_truthy("DYN_SDK_DISABLE_ANSI_LOGGING")
}

/// Check whether to use local timezone for logging timestamps (default is UTC)
/// Set the `DYN_LOG_USE_LOCAL_TZ` environment variable to a [`is_truthy`] value
pub fn use_local_timezone() -> bool {
    env_is_truthy("DYN_LOG_USE_LOCAL_TZ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_with_env_vars() -> Result<()> {
        temp_env::with_vars(
            vec![
                ("DYN_RUNTIME_NUM_WORKER_THREADS", Some("24")),
                ("DYN_RUNTIME_MAX_BLOCKING_THREADS", Some("32")),
            ],
            || {
                let config = RuntimeConfig::from_settings()?;
                assert_eq!(config.num_worker_threads, 24);
                assert_eq!(config.max_blocking_threads, 32);
                Ok(())
            },
        )
    }

    #[test]
    fn test_runtime_config_defaults() -> Result<()> {
        temp_env::with_vars(
            vec![
                ("DYN_RUNTIME_NUM_WORKER_THREADS", None::<&str>),
                ("DYN_RUNTIME_MAX_BLOCKING_THREADS", Some("")),
            ],
            || {
                let config = RuntimeConfig::from_settings()?;

                let default_config = RuntimeConfig::default();
                assert_eq!(config.num_worker_threads, default_config.num_worker_threads);
                assert_eq!(
                    config.max_blocking_threads,
                    default_config.max_blocking_threads
                );
                Ok(())
            },
        )
    }

    #[test]
    fn test_runtime_config_rejects_invalid_thread_count() -> Result<()> {
        temp_env::with_vars(
            vec![
                ("DYN_RUNTIME_NUM_WORKER_THREADS", Some("0")),
                ("DYN_RUNTIME_MAX_BLOCKING_THREADS", Some("0")),
            ],
            || {
                let result = RuntimeConfig::from_settings();
                assert!(result.is_err());
                if let Err(e) = result {
                    assert!(e
                        .to_string()
                        .contains("num_worker_threads: Validation error"));
                    assert!(e
                        .to_string()
                        .contains("max_blocking_threads: Validation error"));
                }
                Ok(())
            },
        )
    }
}
