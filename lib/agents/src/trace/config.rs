// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use super::DEFAULT_TOOL_EVENTS_TOPIC;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_JSONL_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_JSONL_FLUSH_INTERVAL_MS: u64 = 1000;

#[derive(Clone, Debug)]
pub struct AgentTracePolicy {
    pub enabled: bool,
    pub jsonl_path: Option<String>,
    pub capacity: usize,
    pub jsonl_buffer_bytes: usize,
    pub jsonl_flush_interval_ms: u64,
    pub tool_events_enabled: bool,
    pub tool_events_topic: String,
    pub tool_events_namespace: Option<String>,
}

static POLICY: OnceLock<AgentTracePolicy> = OnceLock::new();

fn load_from_env() -> AgentTracePolicy {
    let jsonl_path = std::env::var("DYN_AGENT_TRACE_JSONL")
        .ok()
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty());
    let tool_events_enabled = env_truthy("DYN_AGENT_TRACE_TOOL_EVENTS");
    let tool_events_topic = std::env::var("DYN_AGENT_TRACE_TOOL_EVENTS_TOPIC")
        .ok()
        .map(|topic| topic.trim().to_string())
        .filter(|topic| !topic.is_empty())
        .unwrap_or_else(|| DEFAULT_TOOL_EVENTS_TOPIC.to_string());
    let tool_events_namespace = std::env::var("DYN_AGENT_TRACE_NAMESPACE")
        .ok()
        .map(|namespace| namespace.trim().to_string())
        .filter(|namespace| !namespace.is_empty());
    let capacity = env_usize("DYN_AGENT_TRACE_CAPACITY", DEFAULT_CAPACITY);
    let jsonl_buffer_bytes = env_usize(
        "DYN_AGENT_TRACE_JSONL_BUFFER_BYTES",
        DEFAULT_JSONL_BUFFER_BYTES,
    );
    let jsonl_flush_interval_ms = env_u64(
        "DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS",
        DEFAULT_JSONL_FLUSH_INTERVAL_MS,
    );
    AgentTracePolicy {
        enabled: jsonl_path.is_some() || tool_events_enabled,
        jsonl_path,
        capacity,
        jsonl_buffer_bytes,
        jsonl_flush_interval_ms,
        tool_events_enabled,
        tool_events_topic,
        tool_events_namespace,
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            tracing::warn!(
                key,
                value = %raw,
                default,
                "invalid agent trace numeric env; using default"
            );
            default
        }),
        Err(_) => default,
    }
}

fn env_u64(key: &str, default: u64) -> u64 {
    match std::env::var(key) {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            tracing::warn!(
                key,
                value = %raw,
                default,
                "invalid agent trace numeric env; using default"
            );
            default
        }),
        Err(_) => default,
    }
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub fn init_from_env() -> AgentTracePolicy {
    policy().clone()
}

pub fn policy() -> &'static AgentTracePolicy {
    POLICY.get_or_init(load_from_env)
}

pub fn is_enabled() -> bool {
    policy().enabled
}
