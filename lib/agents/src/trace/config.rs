// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use super::DEFAULT_TOOL_EVENTS_TOPIC;

const DEFAULT_CAPACITY: usize = 1024;

#[derive(Clone, Debug)]
pub struct AgentTracePolicy {
    pub enabled: bool,
    pub jsonl_path: Option<String>,
    pub capacity: usize,
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
    let capacity = match std::env::var("DYN_AGENT_TRACE_CAPACITY") {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            tracing::warn!(
                value = %raw,
                default = DEFAULT_CAPACITY,
                "invalid DYN_AGENT_TRACE_CAPACITY; using default"
            );
            DEFAULT_CAPACITY
        }),
        Err(_) => DEFAULT_CAPACITY,
    };
    AgentTracePolicy {
        enabled: jsonl_path.is_some() || tool_events_enabled,
        jsonl_path,
        capacity,
        tool_events_enabled,
        tool_events_topic,
        tool_events_namespace,
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
