// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::{
    env_is_falsey, env_is_truthy, environment_names::llm::agent_trace as env_agent_trace,
};

use crate::telemetry::parse_sink_names;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_JSONL_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_JSONL_FLUSH_INTERVAL_MS: u64 = 1000;
const DEFAULT_JSONL_GZ_ROLL_BYTES: u64 = 256 * 1024 * 1024;

/// Defaults applied when [`DYN_AGENT_TRACE`] is truthy and the matching
/// per-variable override is unset.
const MASTER_DEFAULT_SINK: &str = "jsonl_gz";
const MASTER_DEFAULT_OUTPUT_PATH: &str = "/tmp/dynamo-agent-trace";
const MASTER_DEFAULT_TOOL_EVENTS_ZMQ_ENDPOINT: &str = "tcp://127.0.0.1:20390";

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
    pub replay_hashes_enabled: bool,
    pub tool_events_zmq_endpoint: Option<String>,
    pub tool_events_zmq_topic: Option<String>,
}

static POLICY: OnceLock<AgentTracePolicy> = OnceLock::new();

fn load_from_env() -> AgentTracePolicy {
    let master_enabled = env_is_truthy(env_agent_trace::DYN_AGENT_TRACE);

    let mut sinks = std::env::var(env_agent_trace::DYN_AGENT_TRACE_SINKS)
        .ok()
        .map(|value| parse_sink_names(&value))
        .unwrap_or_default();
    if sinks.is_empty() && master_enabled {
        sinks = vec![MASTER_DEFAULT_SINK.to_string()];
    }
    let mut output_path = std::env::var(env_agent_trace::DYN_AGENT_TRACE_OUTPUT_PATH)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if output_path.is_none() && master_enabled {
        output_path = Some(MASTER_DEFAULT_OUTPUT_PATH.to_string());
    }
    let mut tool_events_zmq_endpoint =
        std::env::var(env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
    if tool_events_zmq_endpoint.is_none() && master_enabled {
        tool_events_zmq_endpoint = Some(MASTER_DEFAULT_TOOL_EVENTS_ZMQ_ENDPOINT.to_string());
    }
    let tool_events_zmq_topic =
        std::env::var(env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC)
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
    let replay_hashes_enabled = !env_is_falsey(env_agent_trace::DYN_AGENT_TRACE_REPLAY_HASHES);

    AgentTracePolicy {
        enabled: master_enabled || !sinks.is_empty() || tool_events_zmq_endpoint.is_some(),
        sinks,
        output_path,
        capacity,
        jsonl_buffer_bytes,
        jsonl_flush_interval_ms,
        jsonl_gz_roll_bytes,
        jsonl_gz_roll_lines,
        replay_hashes_enabled,
        tool_events_zmq_endpoint,
        tool_events_zmq_topic,
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
    use dynamo_runtime::config::environment_names::llm::agent_trace as env_agent_trace;

    use super::load_from_env;

    #[test]
    #[serial_test::serial]
    fn replay_hashes_default_on() {
        temp_env::with_var_unset(env_agent_trace::DYN_AGENT_TRACE_REPLAY_HASHES, || {
            assert!(load_from_env().replay_hashes_enabled);
        });
    }

    #[test]
    #[serial_test::serial]
    fn replay_hashes_disable_with_falsey_env() {
        temp_env::with_var(
            env_agent_trace::DYN_AGENT_TRACE_REPLAY_HASHES,
            Some("false"),
            || {
                assert!(!load_from_env().replay_hashes_enabled);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn replay_hashes_stay_enabled_with_truthy_env() {
        temp_env::with_var(
            env_agent_trace::DYN_AGENT_TRACE_REPLAY_HASHES,
            Some("1"),
            || {
                assert!(load_from_env().replay_hashes_enabled);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn master_switch_enables_with_default_sink_path_and_endpoint() {
        temp_env::with_vars(
            [
                (env_agent_trace::DYN_AGENT_TRACE, Some("1")),
                (env_agent_trace::DYN_AGENT_TRACE_SINKS, None),
                (env_agent_trace::DYN_AGENT_TRACE_OUTPUT_PATH, None),
                (
                    env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
                    None::<&str>,
                ),
            ],
            || {
                let policy = load_from_env();
                assert!(policy.enabled);
                assert_eq!(policy.sinks, vec!["jsonl_gz".to_string()]);
                assert_eq!(
                    policy.output_path.as_deref(),
                    Some("/tmp/dynamo-agent-trace"),
                );
                assert_eq!(
                    policy.tool_events_zmq_endpoint.as_deref(),
                    Some("tcp://127.0.0.1:20390"),
                );
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn master_switch_yields_to_per_variable_overrides() {
        temp_env::with_vars(
            [
                (env_agent_trace::DYN_AGENT_TRACE, Some("1")),
                (env_agent_trace::DYN_AGENT_TRACE_SINKS, Some("stderr")),
                (
                    env_agent_trace::DYN_AGENT_TRACE_OUTPUT_PATH,
                    Some("/tmp/custom"),
                ),
                (
                    env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
                    Some("tcp://127.0.0.1:9999"),
                ),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(policy.sinks, vec!["stderr".to_string()]);
                assert_eq!(policy.output_path.as_deref(), Some("/tmp/custom"));
                assert_eq!(
                    policy.tool_events_zmq_endpoint.as_deref(),
                    Some("tcp://127.0.0.1:9999"),
                );
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn no_master_switch_keeps_legacy_behavior() {
        temp_env::with_vars(
            [
                (env_agent_trace::DYN_AGENT_TRACE, None::<&str>),
                (env_agent_trace::DYN_AGENT_TRACE_SINKS, None),
                (env_agent_trace::DYN_AGENT_TRACE_OUTPUT_PATH, None),
                (
                    env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
                    None,
                ),
            ],
            || {
                let policy = load_from_env();
                assert!(!policy.enabled);
                assert!(policy.sinks.is_empty());
                assert!(policy.output_path.is_none());
                assert!(policy.tool_events_zmq_endpoint.is_none());
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn master_switch_falsey_value_does_not_enable() {
        temp_env::with_vars(
            [
                (env_agent_trace::DYN_AGENT_TRACE, Some("0")),
                (env_agent_trace::DYN_AGENT_TRACE_SINKS, None),
                (env_agent_trace::DYN_AGENT_TRACE_OUTPUT_PATH, None),
                (
                    env_agent_trace::DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
                    None::<&str>,
                ),
            ],
            || {
                let policy = load_from_env();
                assert!(!policy.enabled);
                assert!(policy.sinks.is_empty());
                assert!(policy.output_path.is_none());
                assert!(policy.tool_events_zmq_endpoint.is_none());
            },
        );
    }
}
