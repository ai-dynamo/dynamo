// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::{
    env_is_truthy, environment_names::llm::request_trace as env_request_trace,
};

use crate::telemetry::parse_sink_names;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_JSONL_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_JSONL_FLUSH_INTERVAL_MS: u64 = 1000;
const DEFAULT_JSONL_GZ_ROLL_BYTES: u64 = 256 * 1024 * 1024;

const DEFAULT_SINK: &str = "jsonl_gz";
const DEFAULT_OUTPUT_PATH: &str = "/tmp/dynamo-request-trace";

#[derive(Clone, Debug)]
pub struct RequestTracePolicy {
    pub enabled: bool,
    pub sinks: Vec<String>,
    pub output_path: Option<String>,
    pub capacity: usize,
    pub jsonl_buffer_bytes: usize,
    pub jsonl_flush_interval_ms: u64,
    pub jsonl_gz_roll_bytes: u64,
    pub jsonl_gz_roll_lines: Option<u64>,
}

static POLICY: OnceLock<RequestTracePolicy> = OnceLock::new();

fn load_from_env() -> RequestTracePolicy {
    let enabled = env_is_truthy(env_request_trace::DYN_REQUEST_TRACE);
    let sinks = std::env::var(env_request_trace::DYN_REQUEST_TRACE_SINKS)
        .ok()
        .map(|value| parse_sink_names(&value))
        .unwrap_or_else(|| {
            if enabled {
                vec![DEFAULT_SINK.to_string()]
            } else {
                Vec::new()
            }
        });
    let output_path = std::env::var(env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| enabled.then(|| DEFAULT_OUTPUT_PATH.to_string()));
    let capacity = std::env::var(env_request_trace::DYN_REQUEST_TRACE_CAPACITY)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CAPACITY);
    let jsonl_buffer_bytes = std::env::var(env_request_trace::DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_JSONL_BUFFER_BYTES);
    let jsonl_flush_interval_ms =
        std::env::var(env_request_trace::DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_JSONL_FLUSH_INTERVAL_MS);
    let jsonl_gz_roll_bytes =
        std::env::var(env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_JSONL_GZ_ROLL_BYTES);
    let jsonl_gz_roll_lines =
        std::env::var(env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0);

    RequestTracePolicy {
        enabled,
        sinks,
        output_path,
        capacity,
        jsonl_buffer_bytes,
        jsonl_flush_interval_ms,
        jsonl_gz_roll_bytes,
        jsonl_gz_roll_lines,
    }
}

pub fn policy() -> &'static RequestTracePolicy {
    POLICY.get_or_init(load_from_env)
}

pub fn is_enabled() -> bool {
    policy().enabled
}

#[cfg(test)]
mod tests {
    use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;

    use super::load_from_env;

    #[test]
    #[serial_test::serial]
    fn master_switch_enables_default_sink_and_path() {
        temp_env::with_vars(
            [
                (env_request_trace::DYN_REQUEST_TRACE, Some("1")),
                (env_request_trace::DYN_REQUEST_TRACE_SINKS, None),
                (env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH, None),
            ],
            || {
                let policy = load_from_env();
                assert!(policy.enabled);
                assert_eq!(policy.sinks, vec!["jsonl_gz".to_string()]);
                assert_eq!(
                    policy.output_path.as_deref(),
                    Some("/tmp/dynamo-request-trace")
                );
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn master_switch_yields_to_overrides() {
        temp_env::with_vars(
            [
                (env_request_trace::DYN_REQUEST_TRACE, Some("1")),
                (
                    env_request_trace::DYN_REQUEST_TRACE_SINKS,
                    Some("jsonl,stderr"),
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH,
                    Some("/tmp/custom-request-trace"),
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES,
                    Some("10"),
                ),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(
                    policy.sinks,
                    vec!["jsonl".to_string(), "stderr".to_string()]
                );
                assert_eq!(
                    policy.output_path.as_deref(),
                    Some("/tmp/custom-request-trace")
                );
                assert_eq!(policy.jsonl_gz_roll_lines, Some(10));
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn disabled_by_default() {
        temp_env::with_vars(
            [
                (env_request_trace::DYN_REQUEST_TRACE, None::<&str>),
                (env_request_trace::DYN_REQUEST_TRACE_SINKS, None),
                (env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH, None),
            ],
            || {
                let policy = load_from_env();
                assert!(!policy.enabled);
                assert!(policy.sinks.is_empty());
                assert!(policy.output_path.is_none());
            },
        );
    }
}
