// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::environment_names::llm::audit as env_audit;

use crate::telemetry::parse_sink_names;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_JSONL_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_JSONL_FLUSH_INTERVAL_MS: u64 = 1000;
const DEFAULT_JSONL_GZ_ROLL_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_SAMPLE_RATE: f32 = 1.0;
const DEFAULT_S3_PREFIX: &str = "dynamo-audit";
const DEFAULT_S3_BATCH_BYTES: usize = 1024 * 1024;
const DEFAULT_S3_ROLL_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_S3_ROLL_INTERVAL_MS: u64 = 60_000;
const DEFAULT_S3_SSE: &str = "AES256";
const DEFAULT_S3_CHANNEL_CAPACITY: usize = 4096;

#[derive(Clone, Debug)]
pub struct AuditPolicy {
    pub enabled: bool,
    pub force_logging: bool,
    pub capacity: usize,
    pub sinks: Vec<String>,
    pub output_path: Option<String>,
    pub jsonl_buffer_bytes: usize,
    pub jsonl_flush_interval_ms: u64,
    pub jsonl_gz_roll_bytes: u64,
    pub jsonl_gz_roll_lines: Option<u64>,
    /// Global head-based sample rate in [0.0, 1.0]. Bypassed by
    /// `force_logging` only. `request.store == true` makes the request
    /// eligible for capture but does NOT bypass sampling.
    pub sample_rate: f32,
    /// Deployment name attached to every audit record. Optional — when
    /// unset, the `deployment` field is absent from records and from S3
    /// object keys.
    pub deployment: Option<String>,
    pub s3_bucket: Option<String>,
    pub s3_prefix: String,
    pub s3_region: Option<String>,
    pub s3_endpoint_url: Option<String>,
    pub s3_batch_bytes: usize,
    pub s3_roll_bytes: u64,
    pub s3_roll_interval_ms: u64,
    pub s3_roll_lines: Option<u64>,
    pub s3_sse: Option<String>,
    pub s3_kms_key_id: Option<String>,
    pub s3_instance_id: Option<String>,
    pub s3_channel_capacity: usize,
}

static POLICY: OnceLock<AuditPolicy> = OnceLock::new();

fn read_string(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Audit is enabled if we have at least one sink
fn load_from_env() -> AuditPolicy {
    let sinks = std::env::var(env_audit::DYN_AUDIT_SINKS)
        .ok()
        .map(|value| parse_sink_names(&value))
        .unwrap_or_default();
    let output_path = read_string(env_audit::DYN_AUDIT_OUTPUT_PATH);
    let capacity = std::env::var(env_audit::DYN_AUDIT_CAPACITY)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_CAPACITY);
    let jsonl_buffer_bytes = std::env::var(env_audit::DYN_AUDIT_JSONL_BUFFER_BYTES)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_JSONL_BUFFER_BYTES);
    let jsonl_flush_interval_ms = std::env::var(env_audit::DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_JSONL_FLUSH_INTERVAL_MS);
    let jsonl_gz_roll_bytes = std::env::var(env_audit::DYN_AUDIT_JSONL_GZ_ROLL_BYTES)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_JSONL_GZ_ROLL_BYTES);
    let jsonl_gz_roll_lines = std::env::var(env_audit::DYN_AUDIT_JSONL_GZ_ROLL_LINES)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0);

    let sample_rate = std::env::var(env_audit::DYN_AUDIT_SAMPLE_RATE)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT_SAMPLE_RATE);
    // Auto-detect the deployment name on Kubernetes when not overridden.
    // The Dynamo operator unconditionally injects DYN_PARENT_DGD_K8S_NAME
    // into every worker pod (see deploy/operator/internal/dynamo/
    // component_common.go), set to the parent DynamoGraphDeployment CR
    // name. Falls back to None on non-K8s deploys, preserving the
    // original behavior.
    let deployment = read_string(env_audit::DYN_AUDIT_DEPLOYMENT)
        .or_else(|| read_string("DYN_PARENT_DGD_K8S_NAME"));

    let s3_bucket = read_string(env_audit::DYN_AUDIT_S3_BUCKET);
    let s3_prefix = read_string(env_audit::DYN_AUDIT_S3_PREFIX)
        .unwrap_or_else(|| DEFAULT_S3_PREFIX.to_string());
    let s3_region = read_string(env_audit::DYN_AUDIT_S3_REGION);
    let s3_endpoint_url = read_string(env_audit::DYN_AUDIT_S3_ENDPOINT_URL);
    let s3_batch_bytes = std::env::var(env_audit::DYN_AUDIT_S3_BATCH_BYTES)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_S3_BATCH_BYTES);
    let s3_roll_bytes = std::env::var(env_audit::DYN_AUDIT_S3_ROLL_BYTES)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_S3_ROLL_BYTES);
    let s3_roll_interval_ms = std::env::var(env_audit::DYN_AUDIT_S3_ROLL_INTERVAL_MS)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_S3_ROLL_INTERVAL_MS);
    let s3_roll_lines = std::env::var(env_audit::DYN_AUDIT_S3_ROLL_LINES)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0);
    // Allow callers to explicitly opt out of SSE by setting the env to
    // "none" (case-insensitive); otherwise fall back to the default.
    let s3_sse = match read_string(env_audit::DYN_AUDIT_S3_SSE) {
        Some(v) if v.eq_ignore_ascii_case("none") => None,
        Some(v) => Some(v),
        None => Some(DEFAULT_S3_SSE.to_string()),
    };
    let s3_kms_key_id = read_string(env_audit::DYN_AUDIT_S3_KMS_KEY_ID);
    let s3_instance_id = read_string(env_audit::DYN_AUDIT_S3_INSTANCE_ID);
    let s3_channel_capacity = std::env::var(env_audit::DYN_AUDIT_S3_CHANNEL_CAPACITY)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_S3_CHANNEL_CAPACITY);

    AuditPolicy {
        enabled: !sinks.is_empty(),
        force_logging: std::env::var(env_audit::DYN_AUDIT_FORCE_LOGGING)
            .ok()
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false),
        capacity,
        sinks,
        output_path,
        jsonl_buffer_bytes,
        jsonl_flush_interval_ms,
        jsonl_gz_roll_bytes,
        jsonl_gz_roll_lines,
        sample_rate,
        deployment,
        s3_bucket,
        s3_prefix,
        s3_region,
        s3_endpoint_url,
        s3_batch_bytes,
        s3_roll_bytes,
        s3_roll_interval_ms,
        s3_roll_lines,
        s3_sse,
        s3_kms_key_id,
        s3_instance_id,
        s3_channel_capacity,
    }
}

/// Returns the singleton audit policy, initialized lazily from env vars on
/// first call. Once initialized, the policy is frozen for the lifetime of
/// the process.
///
/// # Testing
///
/// Because `OnceLock` cannot be reset, **integration tests that need
/// different policy values must live in separate test binaries** (separate
/// files under `tests/`). Unit tests within `audit::handle::tests` work
/// around this by calling `create_handle_with_config(...)` directly
/// instead of going through `policy()`. If you need to test the full
/// `create_handle → policy()` path with different env vars, put each
/// scenario in its own `#[tokio::test]` file or use `cargo test --test
/// <file>` to run them in isolation.
pub fn policy() -> &'static AuditPolicy {
    POLICY.get_or_init(load_from_env)
}
