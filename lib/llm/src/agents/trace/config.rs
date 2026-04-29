// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::environment_names::llm::agent_trace as env_agent_trace;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_JSONL_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_JSONL_FLUSH_INTERVAL_MS: u64 = 1000;
const DEFAULT_JSONL_GZ_ROLL_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct AgentTracePolicy {
    pub enabled: bool,
    pub sinks: Vec<String>,
    pub output_path: Option<String>,
    pub capacity: usize,
    pub jsonl_buffer_bytes: usize,
    pub jsonl_flush_interval_ms: u64,
    pub jsonl_gz_roll_bytes: u64,
    pub jsonl_gz_roll_lines: Option<u64>,
}

static POLICY: OnceLock<AgentTracePolicy> = OnceLock::new();

pub(crate) fn parse_sink_names(value: &str) -> Vec<String> {
    let sinks: Vec<String> = value
        .split(',')
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect();

    if sinks.is_empty() {
        vec!["stderr".to_string()]
    } else {
        sinks
    }
}

fn load_from_env() -> AgentTracePolicy {
    let sinks = std::env::var(env_agent_trace::DYN_AGENT_TRACE_SINKS)
        .ok()
        .map(|value| parse_sink_names(&value))
        .unwrap_or_default();
    let output_path = std::env::var(env_agent_trace::DYN_AGENT_TRACE_OUTPUT_PATH)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let capacity = std::env::var(env_agent_trace::DYN_AGENT_TRACE_CAPACITY)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CAPACITY);
    let jsonl_buffer_bytes = std::env::var(env_agent_trace::DYN_AGENT_TRACE_JSONL_BUFFER_BYTES)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_JSONL_BUFFER_BYTES);
    let jsonl_flush_interval_ms =
        std::env::var(env_agent_trace::DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_JSONL_FLUSH_INTERVAL_MS);
    let jsonl_gz_roll_bytes = std::env::var(env_agent_trace::DYN_AGENT_TRACE_JSONL_GZ_ROLL_BYTES)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_JSONL_GZ_ROLL_BYTES);
    let jsonl_gz_roll_lines = std::env::var(env_agent_trace::DYN_AGENT_TRACE_JSONL_GZ_ROLL_LINES)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0);

    AgentTracePolicy {
        enabled: !sinks.is_empty(),
        sinks,
        output_path,
        capacity,
        jsonl_buffer_bytes,
        jsonl_flush_interval_ms,
        jsonl_gz_roll_bytes,
        jsonl_gz_roll_lines,
    }
}

pub fn policy() -> &'static AgentTracePolicy {
    POLICY.get_or_init(load_from_env)
}

pub fn is_enabled() -> bool {
    policy().enabled
}

#[cfg(test)]
mod tests {
    use super::parse_sink_names;

    #[test]
    fn parse_sink_names_trims_and_normalizes() {
        assert_eq!(
            parse_sink_names(" jsonl, JSONL_GZ, STDERR "),
            vec![
                "jsonl".to_string(),
                "jsonl_gz".to_string(),
                "stderr".to_string()
            ]
        );
    }

    #[test]
    fn parse_sink_names_defaults_empty_value_to_stderr() {
        assert_eq!(parse_sink_names(" , "), vec!["stderr".to_string()]);
    }
}
