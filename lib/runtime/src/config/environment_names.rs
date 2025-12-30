// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Environment variable name constants for centralized management across the codebase
//!
//! This module provides centralized environment variable name constants to ensure
//! consistency and avoid duplication across the codebase, similar to how
//! `prometheus_names.rs` manages metric names.
//!
//! ## Organization
//!
//! Environment variables are organized by functional area:
//! - **Logging**: Log level, configuration, and OTLP tracing
//! - **Runtime**: Tokio runtime configuration and system server settings
//! - **NATS**: NATS client connection and authentication
//! - **ETCD**: ETCD client connection and authentication
//! - **KVBM**: Key-Value Block Manager configuration
//! - **LLM**: Language model inference configuration
//! - **Model**: Model loading and caching
//! - **Worker**: Worker lifecycle and shutdown
//! - **Testing**: Test-specific configuration

/// Logging and tracing environment variables
pub mod logging {
    /// Log level (e.g., "debug", "info", "warn", "error")
    pub const DYN_LOG: &str = "DYN_LOG";

    /// Path to logging configuration file
    pub const DYN_LOGGING_CONFIG_PATH: &str = "DYN_LOGGING_CONFIG_PATH";

    /// Enable JSONL logging format
    pub const DYN_LOGGING_JSONL: &str = "DYN_LOGGING_JSONL";

    /// Disable ANSI terminal colors in logs
    pub const DYN_SDK_DISABLE_ANSI_LOGGING: &str = "DYN_SDK_DISABLE_ANSI_LOGGING";

    /// Use local timezone for logging timestamps (default is UTC)
    pub const DYN_LOG_USE_LOCAL_TZ: &str = "DYN_LOG_USE_LOCAL_TZ";

    /// OTLP (OpenTelemetry Protocol) tracing configuration
    pub mod otlp {
        /// Enable OTLP trace exporting (set to "1" to enable)
        pub const OTEL_EXPORT_ENABLED: &str = "OTEL_EXPORT_ENABLED";

        /// OTLP exporter endpoint URL
        /// Spec: https://opentelemetry.io/docs/specs/otel/protocol/exporter/
        pub const OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT";

        /// Service name for OTLP traces
        pub const OTEL_SERVICE_NAME: &str = "OTEL_SERVICE_NAME";
    }
}

/// Runtime configuration environment variables
///
/// These control the Tokio runtime, system health/metrics server, and worker behavior
pub mod runtime {
    /// Number of async worker threads for Tokio runtime
    pub const DYN_RUNTIME_NUM_WORKER_THREADS: &str = "DYN_RUNTIME_NUM_WORKER_THREADS";

    /// Maximum number of blocking threads for Tokio runtime
    pub const DYN_RUNTIME_MAX_BLOCKING_THREADS: &str = "DYN_RUNTIME_MAX_BLOCKING_THREADS";

    /// System status server configuration
    pub mod system {
        /// Enable system status server for health and metrics endpoints
        /// ⚠️ DEPRECATED: will be removed soon
        pub const DYN_SYSTEM_ENABLED: &str = "DYN_SYSTEM_ENABLED";

        /// System status server host
        pub const DYN_SYSTEM_HOST: &str = "DYN_SYSTEM_HOST";

        /// System status server port
        pub const DYN_SYSTEM_PORT: &str = "DYN_SYSTEM_PORT";

        /// Use endpoint health status for system health
        /// ⚠️ DEPRECATED: No longer used
        pub const DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS: &str =
            "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS";

        /// Starting health status for the system
        pub const DYN_SYSTEM_STARTING_HEALTH_STATUS: &str = "DYN_SYSTEM_STARTING_HEALTH_STATUS";

        /// Health check endpoint path
        pub const DYN_SYSTEM_HEALTH_PATH: &str = "DYN_SYSTEM_HEALTH_PATH";

        /// Liveness check endpoint path
        pub const DYN_SYSTEM_LIVE_PATH: &str = "DYN_SYSTEM_LIVE_PATH";
    }

    /// Compute configuration
    pub mod compute {
        /// Prefix for compute-related environment variables
        pub const PREFIX: &str = "DYN_COMPUTE_";
    }

    /// Canary deployment configuration
    pub mod canary {
        /// Wait time in seconds for canary deployments
        pub const DYN_CANARY_WAIT_TIME: &str = "DYN_CANARY_WAIT_TIME";
    }
}

/// Worker lifecycle environment variables
pub mod worker {
    /// Graceful shutdown timeout in seconds
    pub const DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT: &str = "DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT";
}

/// NATS transport environment variables
pub mod nats {
    /// NATS server address (e.g., "nats://localhost:4222")
    pub const NATS_SERVER: &str = "NATS_SERVER";

    /// NATS authentication environment variables (checked in priority order)
    pub mod auth {
        /// Username for NATS authentication (use with NATS_AUTH_PASSWORD)
        pub const NATS_AUTH_USERNAME: &str = "NATS_AUTH_USERNAME";

        /// Password for NATS authentication (use with NATS_AUTH_USERNAME)
        pub const NATS_AUTH_PASSWORD: &str = "NATS_AUTH_PASSWORD";

        /// Token for NATS authentication
        pub const NATS_AUTH_TOKEN: &str = "NATS_AUTH_TOKEN";

        /// NKey for NATS authentication
        pub const NATS_AUTH_NKEY: &str = "NATS_AUTH_NKEY";

        /// Path to NATS credentials file
        pub const NATS_AUTH_CREDENTIALS_FILE: &str = "NATS_AUTH_CREDENTIALS_FILE";
    }

    /// NATS stream configuration
    pub mod stream {
        /// Maximum age for messages in NATS stream (in seconds)
        pub const DYN_NATS_STREAM_MAX_AGE: &str = "DYN_NATS_STREAM_MAX_AGE";
    }
}

/// ETCD transport environment variables
pub mod etcd {
    /// ETCD endpoints (comma-separated list of URLs)
    pub const ETCD_ENDPOINTS: &str = "ETCD_ENDPOINTS";

    /// ETCD authentication environment variables
    pub mod auth {
        /// Username for ETCD authentication
        pub const ETCD_AUTH_USERNAME: &str = "ETCD_AUTH_USERNAME";

        /// Password for ETCD authentication
        pub const ETCD_AUTH_PASSWORD: &str = "ETCD_AUTH_PASSWORD";

        /// Path to CA certificate for ETCD TLS
        pub const ETCD_AUTH_CA: &str = "ETCD_AUTH_CA";

        /// Path to client certificate for ETCD TLS
        pub const ETCD_AUTH_CLIENT_CERT: &str = "ETCD_AUTH_CLIENT_CERT";

        /// Path to client key for ETCD TLS
        pub const ETCD_AUTH_CLIENT_KEY: &str = "ETCD_AUTH_CLIENT_KEY";
    }
}

/// Key-Value Block Manager (KVBM) environment variables
pub mod kvbm {
    /// Enable KVBM metrics endpoint
    pub const DYN_KVBM_METRICS: &str = "DYN_KVBM_METRICS";

    /// KVBM metrics endpoint port
    pub const DYN_KVBM_METRICS_PORT: &str = "DYN_KVBM_METRICS_PORT";

    /// Enable KVBM recording for debugging
    pub const ENABLE_KVBM_RECORD: &str = "ENABLE_KVBM_RECORD";

    /// Disable disk offload filter
    pub const DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER: &str = "DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER";

    /// CPU cache configuration
    pub mod cpu_cache {
        /// CPU cache size in GB
        pub const DYN_KVBM_CPU_CACHE_GB: &str = "DYN_KVBM_CPU_CACHE_GB";

        /// CPU cache size in number of blocks (override)
        pub const DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS: &str =
            "DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS";
    }

    /// Disk cache configuration
    pub mod disk_cache {
        /// Disk cache size in GB
        pub const DYN_KVBM_DISK_CACHE_GB: &str = "DYN_KVBM_DISK_CACHE_GB";

        /// Disk cache size in number of blocks (override)
        pub const DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS: &str =
            "DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS";
    }

    /// Object storage configuration
    pub mod object_storage {
        /// Enable object storage. Set to "1" to enable.
        pub const DYN_KVBM_OBJECT_ENABLED: &str = "DYN_KVBM_OBJECT_ENABLED";

        /// Bucket name for object storage cache
        /// Supports `{worker_id}` template for per-worker buckets
        /// Example: "kv-cache-{worker_id}"
        pub const DYN_KVBM_OBJECT_BUCKET: &str = "DYN_KVBM_OBJECT_BUCKET";

        /// Endpoint for object storage
        pub const DYN_KVBM_OBJECT_ENDPOINT: &str = "DYN_KVBM_OBJECT_ENDPOINT";

        /// Region for object storage
        pub const DYN_KVBM_OBJECT_REGION: &str = "DYN_KVBM_OBJECT_REGION";

        /// Access key for authentication
        pub const DYN_KVBM_OBJECT_ACCESS_KEY: &str = "DYN_KVBM_OBJECT_ACCESS_KEY";

        /// Secret key for authentication
        pub const DYN_KVBM_OBJECT_SECRET_KEY: &str = "DYN_KVBM_OBJECT_SECRET_KEY";

        /// Number of blocks to store in object storage
        pub const DYN_KVBM_OBJECT_NUM_BLOCKS: &str = "DYN_KVBM_OBJECT_NUM_BLOCKS";
    }
    /// Transfer configuration
    pub mod transfer {
        /// Maximum number of blocks per transfer batch
        pub const DYN_KVBM_TRANSFER_BATCH_SIZE: &str = "DYN_KVBM_TRANSFER_BATCH_SIZE";
    }

    /// KVBM leader (distributed mode) configuration
    pub mod leader {
        /// Timeout in seconds for KVBM leader and worker initialization
        pub const DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS: &str =
            "DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS";

        /// ZMQ host for KVBM leader
        pub const DYN_KVBM_LEADER_ZMQ_HOST: &str = "DYN_KVBM_LEADER_ZMQ_HOST";

        /// ZMQ publish port for KVBM leader
        pub const DYN_KVBM_LEADER_ZMQ_PUB_PORT: &str = "DYN_KVBM_LEADER_ZMQ_PUB_PORT";

        /// ZMQ acknowledgment port for KVBM leader
        pub const DYN_KVBM_LEADER_ZMQ_ACK_PORT: &str = "DYN_KVBM_LEADER_ZMQ_ACK_PORT";
    }

    /// NIXL backend configuration
    pub mod nixl {
        /// Prefix for NIXL backend environment variables
        /// Pattern: DYN_KVBM_NIXL_BACKEND_<backend>=true/false
        /// Example: DYN_KVBM_NIXL_BACKEND_UCX=true
        pub const PREFIX: &str = "DYN_KVBM_NIXL_BACKEND_";
    }
}

/// LLM (Language Model) inference environment variables
pub mod llm {
    /// HTTP body size limit in MB
    pub const DYN_HTTP_BODY_LIMIT_MB: &str = "DYN_HTTP_BODY_LIMIT_MB";

    pub const DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS: &str =
        "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS";

    /// Enable LoRA adapter support (set to "true" to enable)
    pub const DYN_LORA_ENABLED: &str = "DYN_LORA_ENABLED";

    /// LoRA cache directory path
    pub const DYN_LORA_PATH: &str = "DYN_LORA_PATH";

    /// Metrics configuration
    pub mod metrics {
        /// Custom metrics prefix (overrides default "dynamo_frontend")
        pub const DYN_METRICS_PREFIX: &str = "DYN_METRICS_PREFIX";

        /// Histogram bucket configuration (pattern: <PREFIX>_MIN, <PREFIX>_MAX, <PREFIX>_COUNT)
        /// Example: DYN_HISTOGRAM_TTFT_MIN, DYN_HISTOGRAM_TTFT_MAX, DYN_HISTOGRAM_TTFT_COUNT
        pub const HISTOGRAM_PREFIX: &str = "DYN_HISTOGRAM_";
    }
}

/// Model loading and caching environment variables
pub mod model {
    /// Model Express configuration
    pub mod model_express {
        /// Model Express server endpoint URL
        pub const MODEL_EXPRESS_URL: &str = "MODEL_EXPRESS_URL";

        /// Model Express cache path
        pub const MODEL_EXPRESS_CACHE_PATH: &str = "MODEL_EXPRESS_CACHE_PATH";
    }

    /// Hugging Face configuration
    pub mod huggingface {
        /// Hugging Face authentication token
        pub const HF_TOKEN: &str = "HF_TOKEN";

        /// Hugging Face Hub cache directory
        pub const HF_HUB_CACHE: &str = "HF_HUB_CACHE";

        /// Hugging Face home directory
        pub const HF_HOME: &str = "HF_HOME";
    }
}

/// CUDA and GPU environment variables
pub mod cuda {
    /// Path to custom CUDA fatbin file
    ///
    /// Note: build.rs files cannot import this constant at build time,
    /// so they must define local constants with the same value.
    pub const DYNAMO_FATBIN_PATH: &str = "DYNAMO_FATBIN_PATH";
}

/// Build-time environment variables
pub mod build {
    /// Cargo output directory for build artifacts
    ///
    /// Note: This constant cannot be used with the `env!()` macro,
    /// which requires a string literal at compile time.
    /// Build scripts (build.rs) also cannot import this constant.
    pub const OUT_DIR: &str = "OUT_DIR";
}

/// Testing environment variables
pub mod testing {
    /// Enable queued-up request processing in tests
    pub const DYN_QUEUED_UP_PROCESSING: &str = "DYN_QUEUED_UP_PROCESSING";

    /// Soak test run duration (e.g., "3s", "5m")
    pub const DYN_SOAK_RUN_DURATION: &str = "DYN_SOAK_RUN_DURATION";

    /// Soak test batch load size
    pub const DYN_SOAK_BATCH_LOAD: &str = "DYN_SOAK_BATCH_LOAD";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_duplicate_env_var_names() {
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        let vars = [
            // Logging
            logging::DYN_LOG,
            logging::DYN_LOGGING_CONFIG_PATH,
            logging::DYN_LOGGING_JSONL,
            logging::DYN_SDK_DISABLE_ANSI_LOGGING,
            logging::DYN_LOG_USE_LOCAL_TZ,
            logging::otlp::OTEL_EXPORT_ENABLED,
            logging::otlp::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
            logging::otlp::OTEL_SERVICE_NAME,
            // Runtime
            runtime::DYN_RUNTIME_NUM_WORKER_THREADS,
            runtime::DYN_RUNTIME_MAX_BLOCKING_THREADS,
            runtime::system::DYN_SYSTEM_ENABLED,
            runtime::system::DYN_SYSTEM_HOST,
            runtime::system::DYN_SYSTEM_PORT,
            runtime::system::DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS,
            runtime::system::DYN_SYSTEM_STARTING_HEALTH_STATUS,
            runtime::system::DYN_SYSTEM_HEALTH_PATH,
            runtime::system::DYN_SYSTEM_LIVE_PATH,
            runtime::canary::DYN_CANARY_WAIT_TIME,
            // Worker
            worker::DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT,
            // NATS
            nats::NATS_SERVER,
            nats::auth::NATS_AUTH_USERNAME,
            nats::auth::NATS_AUTH_PASSWORD,
            nats::auth::NATS_AUTH_TOKEN,
            nats::auth::NATS_AUTH_NKEY,
            nats::auth::NATS_AUTH_CREDENTIALS_FILE,
            nats::stream::DYN_NATS_STREAM_MAX_AGE,
            // ETCD
            etcd::ETCD_ENDPOINTS,
            etcd::auth::ETCD_AUTH_USERNAME,
            etcd::auth::ETCD_AUTH_PASSWORD,
            etcd::auth::ETCD_AUTH_CA,
            etcd::auth::ETCD_AUTH_CLIENT_CERT,
            etcd::auth::ETCD_AUTH_CLIENT_KEY,
            // KVBM
            kvbm::DYN_KVBM_METRICS,
            kvbm::DYN_KVBM_METRICS_PORT,
            kvbm::ENABLE_KVBM_RECORD,
            kvbm::DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER,
            kvbm::cpu_cache::DYN_KVBM_CPU_CACHE_GB,
            kvbm::cpu_cache::DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS,
            kvbm::disk_cache::DYN_KVBM_DISK_CACHE_GB,
            kvbm::disk_cache::DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS,
            kvbm::leader::DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS,
            kvbm::leader::DYN_KVBM_LEADER_ZMQ_HOST,
            kvbm::leader::DYN_KVBM_LEADER_ZMQ_PUB_PORT,
            kvbm::leader::DYN_KVBM_LEADER_ZMQ_ACK_PORT,
            // LLM
            llm::DYN_HTTP_BODY_LIMIT_MB,
            llm::DYN_LORA_ENABLED,
            llm::DYN_LORA_PATH,
            llm::metrics::DYN_METRICS_PREFIX,
            // Model
            model::model_express::MODEL_EXPRESS_URL,
            model::model_express::MODEL_EXPRESS_CACHE_PATH,
            model::huggingface::HF_TOKEN,
            model::huggingface::HF_HUB_CACHE,
            model::huggingface::HF_HOME,
            // CUDA
            cuda::DYNAMO_FATBIN_PATH,
            // Build
            build::OUT_DIR,
            // Testing
            testing::DYN_QUEUED_UP_PROCESSING,
            testing::DYN_SOAK_RUN_DURATION,
            testing::DYN_SOAK_BATCH_LOAD,
        ];

        for var in &vars {
            if !seen.insert(var) {
                panic!("Duplicate environment variable name: {}", var);
            }
        }
    }

    #[test]
    fn test_naming_conventions() {
        // Dynamo-specific vars should start with DYN_
        assert!(runtime::DYN_RUNTIME_NUM_WORKER_THREADS.starts_with("DYN_"));
        assert!(runtime::system::DYN_SYSTEM_ENABLED.starts_with("DYN_"));
        assert!(kvbm::DYN_KVBM_METRICS.starts_with("DYN_"));
        assert!(worker::DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT.starts_with("DYN_"));

        // NATS vars should start with NATS_
        assert!(nats::NATS_SERVER.starts_with("NATS_"));
        assert!(nats::auth::NATS_AUTH_USERNAME.starts_with("NATS_AUTH_"));

        // ETCD vars should start with ETCD_
        assert!(etcd::ETCD_ENDPOINTS.starts_with("ETCD_"));
        assert!(etcd::auth::ETCD_AUTH_USERNAME.starts_with("ETCD_AUTH_"));

        // OpenTelemetry vars should start with OTEL_
        assert!(logging::otlp::OTEL_EXPORT_ENABLED.starts_with("OTEL_"));
        assert!(logging::otlp::OTEL_SERVICE_NAME.starts_with("OTEL_"));
    }
}
