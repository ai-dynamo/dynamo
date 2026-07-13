// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//! - **TCP Response Stream**: TCP response stream server (CallHome) port and host
//! - **Event Plane**: Event transport selection (NATS)
//! - **KVBM**: Key-Value Block Manager configuration
//! - **LLM**: Language model inference configuration
//! - **Model**: Model loading and caching
//! - **Worker**: Worker lifecycle and shutdown
//! - **Testing**: Test-specific configuration
//! - **Mocker**: Mocker (mock scheduler/KV manager) configuration

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

    /// Enable span event logging (create/close events)
    pub const DYN_LOGGING_SPAN_EVENTS: &str = "DYN_LOGGING_SPAN_EVENTS";

    /// OTLP (OpenTelemetry Protocol) tracing and logging configuration
    pub mod otlp {
        /// Enable OTLP export for traces and logs (set to "1" to enable)
        pub const OTEL_EXPORT_ENABLED: &str = "OTEL_EXPORT_ENABLED";

        /// OTLP exporter transport protocol. Supported values: "grpc", "http/protobuf".
        pub const OTEL_EXPORTER_OTLP_PROTOCOL: &str = "OTEL_EXPORTER_OTLP_PROTOCOL";

        /// OTLP exporter transport protocol for traces. Defaults to OTEL_EXPORTER_OTLP_PROTOCOL.
        pub const OTEL_EXPORTER_OTLP_TRACES_PROTOCOL: &str = "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL";

        /// OTLP exporter transport protocol for logs. Defaults to OTEL_EXPORTER_OTLP_PROTOCOL.
        pub const OTEL_EXPORTER_OTLP_LOGS_PROTOCOL: &str = "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL";

        /// Generic OTLP exporter endpoint URL used when signal-specific endpoints are unset.
        pub const OTEL_EXPORTER_OTLP_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

        /// OTLP exporter endpoint URL for traces
        /// Spec: https://opentelemetry.io/docs/specs/otel/protocol/exporter/
        pub const OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT";

        /// OTLP exporter endpoint URL for logs. Falls back to OTEL_EXPORTER_OTLP_ENDPOINT or the protocol default when unset.
        pub const OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT";

        /// Trace sampling ratio used when set. Example: "0.01" samples roughly 1% of traces.
        pub const OTEL_TRACES_SAMPLE_RATIO: &str = "OTEL_TRACES_SAMPLE_RATIO";

        /// Service name for OTLP traces and logs
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

    /// Maximum time to wait for graceful endpoint drain during runtime shutdown.
    pub const DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS: &str =
        "DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS";

    /// Enable Tokio task poll-time histogram (calls enable_metrics_poll_time_histogram on builder).
    /// Set to "1", "true", or "yes" to enable. Adds ~2× overhead of Instant::now() per task poll.
    pub const DYN_ENABLE_POLL_HISTOGRAM: &str = "DYN_ENABLE_POLL_HISTOGRAM";

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

    /// NATS request/reply timeout in seconds. Unset = async-nats default (10 s).
    pub const DYN_NATS_REQUEST_TIMEOUT_SECS: &str = "DYN_NATS_REQUEST_TIMEOUT_SECS";

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

    /// ETCD lease TTL in seconds (default: 10)
    pub const ETCD_LEASE_TTL: &str = "ETCD_LEASE_TTL";

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

    /// Enable KVBM recording for debugging.
    pub const DYN_KVBM_ENABLE_RECORD: &str = "DYN_KVBM_ENABLE_RECORD";

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

    /// HTTP status code returned when the frontend rejects a request because
    /// all workers are overloaded. Defaults to 529 ("Site is overloaded"); set
    /// to 503 for Service Unavailable retry semantics. Any valid HTTP status
    /// code (100–999) is accepted; an unparseable or out-of-range value falls
    /// back to 529.
    pub const DYN_HTTP_OVERLOAD_STATUS_CODE: &str = "DYN_HTTP_OVERLOAD_STATUS_CODE";

    /// Enable LoRA adapter support (set to "true" to enable)
    pub const DYN_LORA_ENABLED: &str = "DYN_LORA_ENABLED";

    /// LoRA cache directory path
    pub const DYN_LORA_PATH: &str = "DYN_LORA_PATH";

    /// Enable the experimental Anthropic Messages API endpoint (/v1/messages)
    pub const DYN_ENABLE_ANTHROPIC_API: &str = "DYN_ENABLE_ANTHROPIC_API";

    /// Master switch for the `nvext` extension protocol on the frontend.
    /// The protocol is **enabled by default**; this variable disables it.
    /// Truthy values (`1` / `true` / `yes` / `on`, case-insensitive) cause
    /// the frontend to drop `request.nvext` at handler entry, ignore the
    /// routing-override headers (`x-dynamo-worker-instance-id`,
    /// `x-dynamo-prefill-instance-id`, `x-dynamo-dp-rank`,
    /// `x-dynamo-prefill-dp-rank`), and silently ignore the response-side
    /// `extra_fields` opt-in.
    pub const DYN_DISABLE_FRONTEND_NVEXT: &str = "DYN_DISABLE_FRONTEND_NVEXT";

    /// Ignore unknown OpenAI frontend request fields. Unknown fields are dropped,
    /// not handled; known pass-through fields remain type-validated.
    pub const DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS: &str =
        "DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS";

    /// Master switch for the frontend's HTTP admin API surface.
    /// The admin API is **enabled by default**; this variable disables it.
    /// Truthy values (`1` / `true` / `yes` / `on`, case-insensitive) prevent
    /// registration of `GET` / `POST /busy_threshold`. Inference, metrics,
    /// models, health, and liveness routes are unaffected.
    pub const DYN_DISABLE_FRONTEND_ADMIN_API: &str = "DYN_DISABLE_FRONTEND_ADMIN_API";

    /// Strip the Claude Code billing preamble (`x-anthropic-billing-header: ...`)
    /// from the system prompt before forwarding to the target model. The preamble
    /// varies per session and per release, wasting tokens and breaking prompt caching.
    pub const DYN_STRIP_ANTHROPIC_PREAMBLE: &str = "DYN_STRIP_ANTHROPIC_PREAMBLE";

    /// Enable streaming tool call dispatch (`event: tool_call_dispatch` SSE events)
    pub const DYN_ENABLE_STREAMING_TOOL_DISPATCH: &str = "DYN_ENABLE_STREAMING_TOOL_DISPATCH";

    /// Enable streaming reasoning dispatch (`event: reasoning_dispatch` SSE events)
    pub const DYN_ENABLE_STREAMING_REASONING_DISPATCH: &str =
        "DYN_ENABLE_STREAMING_REASONING_DISPATCH";

    /// \[EXPERIMENTAL\] Route supported tool-call families (Qwen3-Coder, DeepSeek-V4)
    /// through the `dynamo-parsers-v2` streaming parser for BOTH the batch and the
    /// streaming path, bypassing the v1 tool-call jail. Off by default; when set, the
    /// v2 parser owns incremental tool-call emission and drops values truncated at EOF.
    pub const DYN_ENABLE_EXPERIMENTAL_PARSERS_V2: &str = "DYN_ENABLE_EXPERIMENTAL_PARSERS_V2";

    /// Backend stream inactivity timeout in seconds.
    ///
    /// When set to a positive integer, the frontend will kill the engine context
    /// and drop the inflight guard if no SSE event is received from the backend
    /// within this many seconds. Acts as a circuit breaker for zombie workers
    /// that hold a live TCP connection but never produce output.
    ///
    /// Set to `0` or leave unset to disable the timeout (default: disabled).
    pub const DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS: &str = "DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS";

    /// Enable the LoRA allocation controller (set to "true" to enable)
    pub const DYN_LORA_ALLOCATION_ENABLED: &str = "DYN_LORA_ALLOCATION_ENABLED";

    /// LoRA allocation algorithm ("hrw", "random", or "mcf")
    pub const DYN_LORA_ALLOCATION_ALGORITHM: &str = "DYN_LORA_ALLOCATION_ALGORITHM";

    /// JSON configuration for the MCF (min-cost flow) placement solver.
    /// Example: '{"candidate_m":16,"gamma_load":2000,"beta_keep":500}'
    /// Omitted fields use defaults. Only relevant when algorithm is "mcf".
    pub const DYN_LORA_MCF_CONFIG: &str = "DYN_LORA_MCF_CONFIG";

    /// LoRA allocation controller recompute interval in seconds
    pub const DYN_LORA_ALLOCATION_TIMESTEP_SECS: &str = "DYN_LORA_ALLOCATION_TIMESTEP_SECS";

    /// Ticks to wait before scaling down a LoRA's replicas
    pub const DYN_LORA_ALLOCATION_SCALE_DOWN_COOLDOWN_TICKS: &str =
        "DYN_LORA_ALLOCATION_SCALE_DOWN_COOLDOWN_TICKS";

    /// Multiplier for the load estimator's rate window relative to the controller timestep.
    pub const DYN_LORA_ALLOCATION_RATE_WINDOW_MULTIPLIER: &str =
        "DYN_LORA_ALLOCATION_RATE_WINDOW_MULTIPLIER";

    /// Number of counter buckets per second in the BucketedRateCounter.
    pub const DYN_LORA_ALLOCATION_BUCKETS_PER_SECOND: &str =
        "DYN_LORA_ALLOCATION_BUCKETS_PER_SECOND";

    /// Load predictor type: "none" (raw counts) or "ema" (exponential moving average).
    pub const DYN_LORA_ALLOCATION_PREDICTOR_TYPE: &str = "DYN_LORA_ALLOCATION_PREDICTOR_TYPE";

    /// EMA smoothing factor (alpha) for the EMA predictor. Range [0.0, 1.0].
    pub const DYN_LORA_ALLOCATION_EMA_ALPHA: &str = "DYN_LORA_ALLOCATION_EMA_ALPHA";

    /// Metrics configuration
    pub mod metrics {
        /// Custom metrics prefix (overrides default "dynamo_frontend")
        pub const DYN_METRICS_PREFIX: &str = "DYN_METRICS_PREFIX";

        /// Histogram bucket configuration (pattern: <PREFIX>_MIN, <PREFIX>_MAX, <PREFIX>_COUNT)
        /// Example: DYN_HISTOGRAM_TTFT_MIN, DYN_HISTOGRAM_TTFT_MAX, DYN_HISTOGRAM_TTFT_COUNT
        pub const HISTOGRAM_PREFIX: &str = "DYN_HISTOGRAM_";
    }

    /// Forward-pass-metrics trace configuration.
    pub mod fpm_trace {
        /// Master switch. Truthy values persist locally produced FPM events.
        pub const DYN_FPM_TRACE: &str = "DYN_FPM_TRACE";

        /// Local gzip JSONL segment prefix. A sanitized producer id is appended
        /// before the segment index so multiple producers do not share files.
        pub const DYN_FPM_OUTPUT_PATH: &str = "DYN_FPM_OUTPUT_PATH";

        /// Capture mode: `sampled` (latest event per DP rank each interval) or
        /// `full` (every event reaching the producer-side trace tap).
        pub const DYN_FPM_MODE: &str = "DYN_FPM_MODE";

        /// Sampling interval in milliseconds when `DYN_FPM_MODE=sampled`.
        pub const DYN_FPM_SAMPLE_INTERVAL_MS: &str = "DYN_FPM_SAMPLE_INTERVAL_MS";

        /// Rotating gzip JSONL threshold in uncompressed bytes.
        pub const DYN_FPM_JSONL_GZ_ROLL_BYTES: &str = "DYN_FPM_JSONL_GZ_ROLL_BYTES";

        /// Maximum number of gzip JSONL segments retained per producer,
        /// including the active segment.
        pub const DYN_FPM_MAX_SEGMENTS: &str = "DYN_FPM_MAX_SEGMENTS";
    }

    /// Deprecated audit payload logging aliases. Prefer `llm::request_trace`.
    pub mod audit {
        /// Deprecated alias for `DYN_REQUEST_TRACE_SINKS`. Legacy values
        /// `jsonl` and `jsonl_gz` map to the request trace `file` sink.
        pub const DYN_AUDIT_SINKS: &str = "DYN_AUDIT_SINKS";

        /// Deprecated migration shim for `DYN_REQUEST_TRACE_RECORDS=request_payload`.
        pub const DYN_AUDIT_FORCE_LOGGING: &str = "DYN_AUDIT_FORCE_LOGGING";

        /// Deprecated alias for `DYN_REQUEST_TRACE_CAPACITY`.
        pub const DYN_AUDIT_CAPACITY: &str = "DYN_AUDIT_CAPACITY";

        /// Deprecated alias for `DYN_REQUEST_TRACE_NATS_SUBJECT`.
        pub const DYN_AUDIT_NATS_SUBJECT: &str = "DYN_AUDIT_NATS_SUBJECT";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_PATH`.
        pub const DYN_AUDIT_OUTPUT_PATH: &str = "DYN_AUDIT_OUTPUT_PATH";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES`.
        pub const DYN_AUDIT_JSONL_BUFFER_BYTES: &str = "DYN_AUDIT_JSONL_BUFFER_BYTES";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS`.
        pub const DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS: &str = "DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_ROLL_BYTES`.
        pub const DYN_AUDIT_JSONL_GZ_ROLL_BYTES: &str = "DYN_AUDIT_JSONL_GZ_ROLL_BYTES";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_ROLL_LINES`.
        pub const DYN_AUDIT_JSONL_GZ_ROLL_LINES: &str = "DYN_AUDIT_JSONL_GZ_ROLL_LINES";

        /// Deprecated alias for `DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES`.
        pub const DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES: &str = "DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES";
    }

    /// Request trace and request payload logging configuration.
    pub mod request_trace {
        /// Master switch. Truthy enables request trace emission.
        pub const DYN_REQUEST_TRACE: &str = "DYN_REQUEST_TRACE";

        /// Request trace sink selection. Comma-separated values: `file`, `stderr`, `nats`, `otel`.
        ///
        /// Legacy values map as follows: `jsonl` => `file` with `jsonl` format,
        /// `jsonl_gz` => `file` with `jsonl_gz` format, `stderr` => `stderr`,
        /// `nats` => `nats`, and `otel` => `otel`.
        pub const DYN_REQUEST_TRACE_SINKS: &str = "DYN_REQUEST_TRACE_SINKS";

        /// Local output path for request trace file records.
        ///
        /// With `DYN_REQUEST_TRACE_FILE_FORMAT=jsonl`, this is the literal JSONL
        /// path. With `jsonl_gz`, this is the segment prefix used to derive
        /// `<prefix>.<index>.jsonl.gz` files.
        pub const DYN_REQUEST_TRACE_FILE_PATH: &str = "DYN_REQUEST_TRACE_FILE_PATH";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_PATH`.
        pub const DYN_REQUEST_TRACE_OUTPUT_PATH: &str = "DYN_REQUEST_TRACE_OUTPUT_PATH";

        /// Request trace file record format. Supported values: `jsonl`, `jsonl_gz`.
        pub const DYN_REQUEST_TRACE_FILE_FORMAT: &str = "DYN_REQUEST_TRACE_FILE_FORMAT";

        /// In-process trace bus capacity.
        pub const DYN_REQUEST_TRACE_CAPACITY: &str = "DYN_REQUEST_TRACE_CAPACITY";

        /// Request trace record selection. Comma-separated values: `request_end`,
        /// `request_payload`, `tool`.
        pub const DYN_REQUEST_TRACE_RECORDS: &str = "DYN_REQUEST_TRACE_RECORDS";

        /// NATS subject the request trace sink publishes to.
        pub const DYN_REQUEST_TRACE_NATS_SUBJECT: &str = "DYN_REQUEST_TRACE_NATS_SUBJECT";

        /// Maximum serialized OTEL payload bytes. Oversized request payload
        /// records emit an incomplete marker payload instead of the full request/response.
        pub const DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES: &str =
            "DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES";

        /// Request trace file sink buffer size in bytes.
        pub const DYN_REQUEST_TRACE_FILE_BUFFER_BYTES: &str = "DYN_REQUEST_TRACE_FILE_BUFFER_BYTES";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_BUFFER_BYTES`.
        pub const DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES: &str =
            "DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES";

        /// Request trace file sink periodic flush interval in milliseconds.
        pub const DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS: &str =
            "DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS`.
        pub const DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS: &str =
            "DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS";

        /// Gzip file sink roll threshold in uncompressed bytes.
        pub const DYN_REQUEST_TRACE_FILE_ROLL_BYTES: &str = "DYN_REQUEST_TRACE_FILE_ROLL_BYTES";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_ROLL_BYTES`.
        pub const DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES: &str =
            "DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES";

        /// Gzip file sink roll threshold in record lines.
        pub const DYN_REQUEST_TRACE_FILE_ROLL_LINES: &str = "DYN_REQUEST_TRACE_FILE_ROLL_LINES";

        /// Deprecated alias for `DYN_REQUEST_TRACE_FILE_ROLL_LINES`.
        pub const DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES: &str =
            "DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES";

        /// Local ZMQ PULL endpoint Dynamo binds for harness tool events.
        pub const DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT: &str =
            "DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT";

        /// First-frame ZMQ topic filter override for harness tool events.
        pub const DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC: &str =
            "DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC";

        /// Comma/whitespace-separated allowlist of HTTP request header names to
        /// record in request payload records. Unset/empty captures none. Values
        /// are recorded unredacted; avoid credential-bearing headers.
        pub const DYN_REQUEST_TRACE_HTTP_HEADER_CAPTURE_LIST: &str =
            "DYN_REQUEST_TRACE_HTTP_HEADER_CAPTURE_LIST";
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

        /// Disable shared-storage mode for the Model Express client. When set,
        /// the client streams model files from the server over gRPC instead of
        /// relying on a shared filesystem path. Required when the Model Express
        /// server and worker pods do not share a filesystem (e.g. RWO PVCs,
        /// cross-namespace deployments). Set to "1" or "true" to enable.
        pub const MODEL_EXPRESS_NO_SHARED_STORAGE: &str = "MODEL_EXPRESS_NO_SHARED_STORAGE";
    }

    /// Hugging Face configuration
    pub mod huggingface {
        /// Hugging Face authentication token
        pub const HF_TOKEN: &str = "HF_TOKEN";

        /// Hugging Face Hub cache directory
        pub const HF_HUB_CACHE: &str = "HF_HUB_CACHE";

        /// Hugging Face home directory
        pub const HF_HOME: &str = "HF_HOME";

        /// Offline mode - skip API calls when model is cached
        /// Set to "1" or "true" to enable
        pub const HF_HUB_OFFLINE: &str = "HF_HUB_OFFLINE";
    }
}

/// KV Router configuration environment variables
pub mod router {
    /// Scale applied to adjusted prompt-side prefill load after overlap/cache-hit credits.
    pub const DYN_ROUTER_PREFILL_LOAD_SCALE: &str = "DYN_ROUTER_PREFILL_LOAD_SCALE";

    /// Queue threshold fraction for prefill token capacity.
    /// When set, requests are queued if all workers exceed this fraction of max_num_batched_tokens.
    pub const DYN_ROUTER_QUEUE_THRESHOLD: &str = "DYN_ROUTER_QUEUE_THRESHOLD";

    /// Scheduling policy for the router queue ("fcfs" or "wspt").
    pub const DYN_ROUTER_QUEUE_POLICY: &str = "DYN_ROUTER_QUEUE_POLICY";
    pub const DYN_ROUTER_POLICY_CONFIG: &str = "DYN_ROUTER_POLICY_CONFIG";
}

/// Request plane transport environment variables
pub mod request_plane {
    /// Request plane payload codec selection: "json" or "msgpack".
    /// JSON is the compatibility default.
    pub const DYN_REQUEST_PLANE_CODEC: &str = "DYN_REQUEST_PLANE_CODEC";
}

/// TCP response stream server (CallHome listener) environment variables
pub mod tcp_response_stream {
    /// Port for the TCP response stream server.
    /// If unset or 0, the OS assigns a free ephemeral port.
    pub const DYN_TCP_RESPONSE_STREAM_PORT: &str = "DYN_TCP_RESPONSE_STREAM_PORT";

    /// Host/interface for the TCP response stream server.
    /// If unset, the server auto-detects a routable local IP.
    pub const DYN_TCP_RESPONSE_STREAM_HOST: &str = "DYN_TCP_RESPONSE_STREAM_HOST";

    /// TCP request-plane TLS configuration
    pub mod tls {
        /// Path to the PEM certificate used by the TCP server.
        /// When set together with DYN_TCP_TLS_KEY_PATH, TLS is enabled on the
        /// TCP server. To enable TLS on the client side, also set
        /// DYN_TCP_TLS_CA_CERT_PATH (or DYN_TCP_TLS_INSECURE for dev).
        pub const DYN_TCP_TLS_CERT_PATH: &str = "DYN_TCP_TLS_CERT_PATH";

        /// Path to the PEM private key for the TCP server certificate.
        pub const DYN_TCP_TLS_KEY_PATH: &str = "DYN_TCP_TLS_KEY_PATH";

        /// Path to the PEM CA certificate used by TCP clients to verify the server.
        /// Required on the client side when the server uses a self-signed or internal CA.
        pub const DYN_TCP_TLS_CA_CERT_PATH: &str = "DYN_TCP_TLS_CA_CERT_PATH";

        /// Disable TLS certificate verification on the TCP client. Set to "true" to skip.
        /// WARNING: Only for local development. Never use in production.
        pub const DYN_TCP_TLS_INSECURE: &str = "DYN_TCP_TLS_INSECURE";

        /// Override the TLS server name (SNI) used by TCP clients when verifying the
        /// server certificate. When unset, the hostname extracted from the connection
        /// address is used. Useful when connecting by IP to a server whose certificate
        /// uses a DNS SAN.
        pub const DYN_TCP_TLS_SERVER_NAME: &str = "DYN_TCP_TLS_SERVER_NAME";

        /// TLS handshake timeout in seconds (default: 3).
        pub const DYN_TCP_TLS_HANDSHAKE_TIMEOUT_SECS: &str = "DYN_TCP_TLS_HANDSHAKE_TIMEOUT_SECS";
    }
}

/// Event Plane transport environment variables
pub mod event_plane {
    /// Event transport selection: "zmq" or "nats".
    ///
    /// When unset the default depends on the discovery backend:
    /// - `file` / `mem` backends: defaults to `zmq` (no external services required).
    /// - `etcd` / `kubernetes` backends: defaults to `nats`.
    ///
    /// Set this explicitly to override the context-aware default.
    pub const DYN_EVENT_PLANE: &str = "DYN_EVENT_PLANE";

    /// Event plane codec selection: "json" or "msgpack".
    pub const DYN_EVENT_PLANE_CODEC: &str = "DYN_EVENT_PLANE_CODEC";

    /// Bounded capacity of the direct ZMQ event-subscriber's merged event channel.
    ///
    /// Many peer publishers (e.g. every other frontend under replica-sync) feed
    /// this single-consumer channel; an unbounded channel grows RSS without limit
    /// when the consumer can't keep up. When the channel is full, new events are
    /// dropped — the event plane is already best-effort/lossy (ZMQ RCVHWM), so a
    /// dropped event costs routing-estimate freshness, not correctness.
    /// Default: 100_000 (matches ZMQ_RCVHWM). Applies only to the direct ZMQ
    /// subscriber path.
    pub const DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY: &str =
        "DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY";
}

/// ZMQ Broker environment variables
pub mod zmq_broker {
    /// Explicit ZMQ broker URL (takes precedence over discovery)
    /// Format: "xsub=<url1>[;<url2>...] , xpub=<url1>[;<url2>...]"
    /// Example: "xsub=tcp://broker:5555 , xpub=tcp://broker:5556"
    pub const DYN_ZMQ_BROKER_URL: &str = "DYN_ZMQ_BROKER_URL";

    /// Enable ZMQ broker discovery mode
    pub const DYN_ZMQ_BROKER_ENABLED: &str = "DYN_ZMQ_BROKER_ENABLED";

    /// XSUB bind address (broker binary only)
    pub const ZMQ_BROKER_XSUB_BIND: &str = "ZMQ_BROKER_XSUB_BIND";

    /// XPUB bind address (broker binary only)
    pub const ZMQ_BROKER_XPUB_BIND: &str = "ZMQ_BROKER_XPUB_BIND";

    /// Namespace for broker discovery registration
    pub const ZMQ_BROKER_NAMESPACE: &str = "ZMQ_BROKER_NAMESPACE";
}

/// Discovery environment variables
pub mod discovery {
    /// Discovery backend: "kubernetes" or "etcd" (default)
    pub const DYN_DISCOVERY_BACKEND: &str = "DYN_DISCOVERY_BACKEND";

    /// Kube discovery mode: "pod" (default) or "container" (each container registers independently)
    pub const DYN_KUBE_DISCOVERY_MODE: &str = "DYN_KUBE_DISCOVERY_MODE";
}

/// CUDA and GPU environment variables
pub mod cuda {
    /// Path to custom CUDA fatbin file.
    ///
    /// Note: build.rs files cannot import this constant at build time,
    /// so they must define local constants with the same value.
    pub const DYN_FATBIN_PATH: &str = "DYN_FATBIN_PATH";
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

/// Mocker (mock scheduler/KV manager) environment variables
pub mod mocker {
    /// Enable structured KV cache allocation/eviction trace logs (set to "1" or "true" to enable)
    pub const DYN_MOCKER_KV_CACHE_TRACE: &str = "DYN_MOCKER_KV_CACHE_TRACE";

    /// Use the original direct() code path in the mocker request dispatch.
    ///
    /// This path is race-prone during startup; prefer leaving it unset unless you are
    /// explicitly trying to reproduce the original behavior.
    pub const DYN_MOCKER_SYNC_DIRECT: &str = "DYN_MOCKER_SYNC_DIRECT";
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
            logging::DYN_LOGGING_SPAN_EVENTS,
            logging::otlp::OTEL_EXPORT_ENABLED,
            logging::otlp::OTEL_EXPORTER_OTLP_PROTOCOL,
            logging::otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
            logging::otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL,
            logging::otlp::OTEL_EXPORTER_OTLP_ENDPOINT,
            logging::otlp::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
            logging::otlp::OTEL_SERVICE_NAME,
            logging::otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT,
            logging::otlp::OTEL_TRACES_SAMPLE_RATIO,
            // Runtime
            runtime::DYN_RUNTIME_NUM_WORKER_THREADS,
            runtime::DYN_RUNTIME_MAX_BLOCKING_THREADS,
            runtime::DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
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
            nats::DYN_NATS_REQUEST_TIMEOUT_SECS,
            nats::auth::NATS_AUTH_USERNAME,
            nats::auth::NATS_AUTH_PASSWORD,
            nats::auth::NATS_AUTH_TOKEN,
            nats::auth::NATS_AUTH_NKEY,
            nats::auth::NATS_AUTH_CREDENTIALS_FILE,
            nats::stream::DYN_NATS_STREAM_MAX_AGE,
            // ETCD
            etcd::ETCD_ENDPOINTS,
            etcd::ETCD_LEASE_TTL,
            etcd::auth::ETCD_AUTH_USERNAME,
            etcd::auth::ETCD_AUTH_PASSWORD,
            etcd::auth::ETCD_AUTH_CA,
            etcd::auth::ETCD_AUTH_CLIENT_CERT,
            etcd::auth::ETCD_AUTH_CLIENT_KEY,
            // KVBM
            kvbm::DYN_KVBM_METRICS,
            kvbm::DYN_KVBM_METRICS_PORT,
            kvbm::DYN_KVBM_ENABLE_RECORD,
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
            llm::DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
            llm::DYN_HTTP_OVERLOAD_STATUS_CODE,
            llm::DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS,
            llm::DYN_LORA_ENABLED,
            llm::DYN_LORA_PATH,
            llm::DYN_ENABLE_ANTHROPIC_API,
            llm::DYN_DISABLE_FRONTEND_NVEXT,
            llm::DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS,
            llm::DYN_DISABLE_FRONTEND_ADMIN_API,
            llm::DYN_STRIP_ANTHROPIC_PREAMBLE,
            llm::DYN_ENABLE_STREAMING_TOOL_DISPATCH,
            llm::DYN_ENABLE_STREAMING_REASONING_DISPATCH,
            llm::DYN_ENABLE_EXPERIMENTAL_PARSERS_V2,
            llm::DYN_LORA_ALLOCATION_ENABLED,
            llm::DYN_LORA_ALLOCATION_ALGORITHM,
            llm::DYN_LORA_ALLOCATION_TIMESTEP_SECS,
            llm::DYN_LORA_ALLOCATION_SCALE_DOWN_COOLDOWN_TICKS,
            llm::DYN_LORA_ALLOCATION_RATE_WINDOW_MULTIPLIER,
            llm::DYN_LORA_ALLOCATION_BUCKETS_PER_SECOND,
            llm::DYN_LORA_ALLOCATION_PREDICTOR_TYPE,
            llm::DYN_LORA_ALLOCATION_EMA_ALPHA,
            llm::DYN_LORA_MCF_CONFIG,
            llm::metrics::DYN_METRICS_PREFIX,
            llm::audit::DYN_AUDIT_SINKS,
            llm::audit::DYN_AUDIT_FORCE_LOGGING,
            llm::audit::DYN_AUDIT_CAPACITY,
            llm::audit::DYN_AUDIT_NATS_SUBJECT,
            llm::audit::DYN_AUDIT_OUTPUT_PATH,
            llm::audit::DYN_AUDIT_JSONL_BUFFER_BYTES,
            llm::audit::DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS,
            llm::audit::DYN_AUDIT_JSONL_GZ_ROLL_BYTES,
            llm::audit::DYN_AUDIT_JSONL_GZ_ROLL_LINES,
            llm::request_trace::DYN_REQUEST_TRACE,
            llm::request_trace::DYN_REQUEST_TRACE_SINKS,
            llm::request_trace::DYN_REQUEST_TRACE_FILE_PATH,
            llm::request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH,
            llm::request_trace::DYN_REQUEST_TRACE_FILE_FORMAT,
            llm::request_trace::DYN_REQUEST_TRACE_CAPACITY,
            llm::request_trace::DYN_REQUEST_TRACE_RECORDS,
            llm::request_trace::DYN_REQUEST_TRACE_NATS_SUBJECT,
            llm::request_trace::DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES,
            llm::request_trace::DYN_REQUEST_TRACE_FILE_BUFFER_BYTES,
            llm::request_trace::DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES,
            llm::request_trace::DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS,
            llm::request_trace::DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS,
            llm::request_trace::DYN_REQUEST_TRACE_FILE_ROLL_BYTES,
            llm::request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES,
            llm::request_trace::DYN_REQUEST_TRACE_FILE_ROLL_LINES,
            llm::request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES,
            llm::request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
            llm::request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC,
            llm::request_trace::DYN_REQUEST_TRACE_HTTP_HEADER_CAPTURE_LIST,
            llm::audit::DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES,
            // Model
            model::model_express::MODEL_EXPRESS_URL,
            model::model_express::MODEL_EXPRESS_CACHE_PATH,
            model::model_express::MODEL_EXPRESS_NO_SHARED_STORAGE,
            model::huggingface::HF_TOKEN,
            model::huggingface::HF_HUB_CACHE,
            model::huggingface::HF_HOME,
            model::huggingface::HF_HUB_OFFLINE,
            // Router
            router::DYN_ROUTER_PREFILL_LOAD_SCALE,
            router::DYN_ROUTER_QUEUE_THRESHOLD,
            router::DYN_ROUTER_QUEUE_POLICY,
            router::DYN_ROUTER_POLICY_CONFIG,
            request_plane::DYN_REQUEST_PLANE_CODEC,
            // TCP Response Stream
            tcp_response_stream::DYN_TCP_RESPONSE_STREAM_PORT,
            tcp_response_stream::DYN_TCP_RESPONSE_STREAM_HOST,
            tcp_response_stream::tls::DYN_TCP_TLS_CERT_PATH,
            tcp_response_stream::tls::DYN_TCP_TLS_KEY_PATH,
            tcp_response_stream::tls::DYN_TCP_TLS_CA_CERT_PATH,
            tcp_response_stream::tls::DYN_TCP_TLS_INSECURE,
            tcp_response_stream::tls::DYN_TCP_TLS_SERVER_NAME,
            tcp_response_stream::tls::DYN_TCP_TLS_HANDSHAKE_TIMEOUT_SECS,
            // Event Plane
            event_plane::DYN_EVENT_PLANE,
            event_plane::DYN_EVENT_PLANE_CODEC,
            event_plane::DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY,
            // ZMQ Broker
            zmq_broker::DYN_ZMQ_BROKER_URL,
            zmq_broker::DYN_ZMQ_BROKER_ENABLED,
            zmq_broker::ZMQ_BROKER_XSUB_BIND,
            zmq_broker::ZMQ_BROKER_XPUB_BIND,
            zmq_broker::ZMQ_BROKER_NAMESPACE,
            // Discovery
            discovery::DYN_DISCOVERY_BACKEND,
            discovery::DYN_KUBE_DISCOVERY_MODE,
            // CUDA
            cuda::DYN_FATBIN_PATH,
            // Build
            build::OUT_DIR,
            // Mocker
            mocker::DYN_MOCKER_KV_CACHE_TRACE,
            mocker::DYN_MOCKER_SYNC_DIRECT,
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
        assert!(runtime::DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS.starts_with("DYN_"));
        assert!(runtime::system::DYN_SYSTEM_ENABLED.starts_with("DYN_"));
        assert!(kvbm::DYN_KVBM_METRICS.starts_with("DYN_"));
        assert!(worker::DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT.starts_with("DYN_"));

        // NATS vars should start with NATS_
        assert!(nats::NATS_SERVER.starts_with("NATS_"));
        assert!(nats::auth::NATS_AUTH_USERNAME.starts_with("NATS_AUTH_"));

        // ETCD vars should start with ETCD_
        assert!(etcd::ETCD_ENDPOINTS.starts_with("ETCD_"));
        assert!(etcd::ETCD_LEASE_TTL.starts_with("ETCD_"));
        assert!(etcd::auth::ETCD_AUTH_USERNAME.starts_with("ETCD_AUTH_"));

        // OpenTelemetry vars should start with OTEL_
        assert!(logging::otlp::OTEL_EXPORT_ENABLED.starts_with("OTEL_"));
        assert!(logging::otlp::OTEL_EXPORTER_OTLP_ENDPOINT.starts_with("OTEL_"));
        assert!(logging::otlp::OTEL_SERVICE_NAME.starts_with("OTEL_"));
    }
}
