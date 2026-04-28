// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{fmt::Display, str::FromStr, sync::OnceLock};

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
    pub tool_events_zmq_endpoint: Option<String>,
    pub tool_events_zmq_topic: Option<String>,
}

static POLICY: OnceLock<AgentTracePolicy> = OnceLock::new();

fn load_from_env() -> AgentTracePolicy {
    let jsonl_path = non_empty_env("DYN_AGENT_TRACE_JSONL");
    let tool_events_topic = non_empty_env("DYN_AGENT_TRACE_TOOL_EVENTS_TOPIC")
        .unwrap_or_else(|| DEFAULT_TOOL_EVENTS_TOPIC.to_string());
    let tool_events_namespace = non_empty_env("DYN_AGENT_TRACE_NAMESPACE");
    let tool_events_zmq_endpoint = non_empty_env("DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT");
    let tool_events_zmq_topic = non_empty_env("DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC");
    let tool_events_enabled =
        env_truthy("DYN_AGENT_TRACE_TOOL_EVENTS") || tool_events_zmq_endpoint.is_some();
    let capacity = env_parse("DYN_AGENT_TRACE_CAPACITY", DEFAULT_CAPACITY);
    let jsonl_buffer_bytes = env_parse(
        "DYN_AGENT_TRACE_JSONL_BUFFER_BYTES",
        DEFAULT_JSONL_BUFFER_BYTES,
    );
    let jsonl_flush_interval_ms = env_parse(
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
        tool_events_zmq_endpoint,
        tool_events_zmq_topic,
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_parse<T>(key: &str, default: T) -> T
where
    T: Copy + Display + FromStr,
{
    match std::env::var(key) {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            tracing::warn!(
                key,
                value = %raw,
                default = %default,
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

pub fn policy() -> &'static AgentTracePolicy {
    POLICY.get_or_init(load_from_env)
}

pub fn is_enabled() -> bool {
    policy().enabled
}
