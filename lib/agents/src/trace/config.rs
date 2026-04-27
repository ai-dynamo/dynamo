// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct AgentTracePolicy {
    pub enabled: bool,
    pub jsonl_path: Option<String>,
    pub capacity: usize,
}

static POLICY: OnceLock<AgentTracePolicy> = OnceLock::new();

pub fn init_from_env() -> AgentTracePolicy {
    let jsonl_path = std::env::var("DYN_AGENT_TRACE_JSONL").ok();
    let capacity = std::env::var("DYN_AGENT_TRACE_CAPACITY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    AgentTracePolicy {
        enabled: jsonl_path.is_some(),
        jsonl_path,
        capacity,
    }
}

pub fn policy() -> &'static AgentTracePolicy {
    POLICY.get_or_init(init_from_env)
}

pub fn is_enabled() -> bool {
    policy().enabled
}
