// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

const DEFAULT_CAPACITY: usize = 1024;

#[derive(Clone, Debug)]
pub struct AgentTracePolicy {
    pub enabled: bool,
    pub jsonl_path: Option<String>,
    pub capacity: usize,
}

static POLICY: OnceLock<AgentTracePolicy> = OnceLock::new();

fn load_from_env() -> AgentTracePolicy {
    let jsonl_path = std::env::var("DYN_AGENT_TRACE_JSONL")
        .ok()
        .map(|path| path.trim().to_string())
        .filter(|path| !path.is_empty());
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
        enabled: jsonl_path.is_some(),
        jsonl_path,
        capacity,
    }
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
