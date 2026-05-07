// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::environment_names::llm::audit as env_audit;

use crate::telemetry::parse_sink_names;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_JSONL_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_JSONL_FLUSH_INTERVAL_MS: u64 = 1000;
const DEFAULT_JSONL_GZ_ROLL_BYTES: u64 = 256 * 1024 * 1024;

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
}

static POLICY: OnceLock<AuditPolicy> = OnceLock::new();

/// Audit is enabled if we have at least one sink
fn load_from_env() -> AuditPolicy {
    let sinks = std::env::var(env_audit::DYN_AUDIT_SINKS)
        .ok()
        .map(|value| parse_sink_names(&value))
        .unwrap_or_default();
    let output_path = std::env::var(env_audit::DYN_AUDIT_OUTPUT_PATH)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
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
    }
}

pub fn policy() -> &'static AuditPolicy {
    POLICY.get_or_init(load_from_env)
}
