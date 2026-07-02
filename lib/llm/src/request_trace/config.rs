// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::env_is_truthy;
use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;

use crate::telemetry::parse_sink_names;

use super::DEFAULT_TOOL_EVENTS_TOPIC;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_FILE_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_FILE_FLUSH_INTERVAL_MS: u64 = 1000;
const DEFAULT_FILE_ROLL_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_FILE_PATH: &str = "/tmp/dynamo-request-trace";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestTraceDestination {
    File,
    Stderr,
}

impl RequestTraceDestination {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Stderr => "stderr",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestTraceFileFormat {
    Jsonl,
}

impl RequestTraceFileFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jsonl => "jsonl",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestTraceFileCompression {
    None,
    Gzip,
}

impl RequestTraceFileCompression {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Gzip => "gzip",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RequestTracePolicy {
    pub enabled: bool,
    pub destinations: Vec<RequestTraceDestination>,
    pub file_path: Option<String>,
    pub file_format: RequestTraceFileFormat,
    pub file_compression: RequestTraceFileCompression,
    pub capacity: usize,
    pub file_buffer_bytes: usize,
    pub file_flush_interval_ms: u64,
    pub file_roll_bytes: u64,
    pub file_roll_lines: Option<u64>,
    pub tool_events_zmq_endpoint: Option<String>,
    pub tool_events_zmq_topic: Option<String>,
}

impl RequestTracePolicy {
    pub fn destination_names(&self) -> Vec<&'static str> {
        self.destinations
            .iter()
            .map(|destination| destination.as_str())
            .collect()
    }
}

static POLICY: OnceLock<RequestTracePolicy> = OnceLock::new();

fn load_from_env() -> RequestTracePolicy {
    let enabled = env_is_truthy(env_request_trace::DYN_REQUEST_TRACE);
    let (destinations, legacy_file_compression) = load_destinations(enabled);
    let has_file_destination = destinations.contains(&RequestTraceDestination::File);
    let file_path = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_FILE_PATH)
        .or_else(|| env_trimmed(env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH))
        .or_else(|| (enabled && has_file_destination).then(|| DEFAULT_FILE_PATH.to_string()));
    let file_format = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_FILE_FORMAT)
        .as_deref()
        .map(parse_file_format)
        .unwrap_or(RequestTraceFileFormat::Jsonl);
    let file_compression = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_FILE_COMPRESSION)
        .as_deref()
        .map(parse_file_compression)
        .unwrap_or_else(|| legacy_file_compression.unwrap_or(RequestTraceFileCompression::Gzip));
    let capacity = std::env::var(env_request_trace::DYN_REQUEST_TRACE_CAPACITY)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CAPACITY);
    let file_buffer_bytes = env_usize(
        env_request_trace::DYN_REQUEST_TRACE_FILE_BUFFER_BYTES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES,
    )
    .unwrap_or(DEFAULT_FILE_BUFFER_BYTES);
    let file_flush_interval_ms = env_u64(
        env_request_trace::DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS,
    )
    .unwrap_or(DEFAULT_FILE_FLUSH_INTERVAL_MS);
    let file_roll_bytes = env_u64(
        env_request_trace::DYN_REQUEST_TRACE_FILE_ROLL_BYTES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES,
    )
    .filter(|value| *value > 0)
    .unwrap_or(DEFAULT_FILE_ROLL_BYTES);
    let file_roll_lines = env_u64(
        env_request_trace::DYN_REQUEST_TRACE_FILE_ROLL_LINES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES,
    )
    .filter(|value| *value > 0);
    let tool_events_zmq_endpoint =
        std::env::var(env_request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
    let tool_events_zmq_topic = tool_events_zmq_endpoint.as_ref().map(|_| {
        std::env::var(env_request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_TOOL_EVENTS_TOPIC.to_string())
    });

    RequestTracePolicy {
        enabled,
        destinations,
        file_path,
        file_format,
        file_compression,
        capacity,
        file_buffer_bytes,
        file_flush_interval_ms,
        file_roll_bytes,
        file_roll_lines,
        tool_events_zmq_endpoint,
        tool_events_zmq_topic,
    }
}

fn load_destinations(
    enabled: bool,
) -> (
    Vec<RequestTraceDestination>,
    Option<RequestTraceFileCompression>,
) {
    if let Ok(value) = std::env::var(env_request_trace::DYN_REQUEST_TRACE_DESTINATIONS) {
        parse_destination_names(&value)
    } else if let Ok(value) = std::env::var(env_request_trace::DYN_REQUEST_TRACE_SINKS) {
        parse_destination_names(&value)
    } else if enabled {
        (vec![RequestTraceDestination::File], None)
    } else {
        (Vec::new(), None)
    }
}

fn parse_destination_names(
    value: &str,
) -> (
    Vec<RequestTraceDestination>,
    Option<RequestTraceFileCompression>,
) {
    let mut destinations = Vec::new();
    let mut legacy_jsonl = false;
    let mut legacy_jsonl_gz = false;

    for name in parse_sink_names(value) {
        match name.as_str() {
            "file" => push_destination(&mut destinations, RequestTraceDestination::File),
            "stderr" => push_destination(&mut destinations, RequestTraceDestination::Stderr),
            "jsonl" => {
                legacy_jsonl = true;
                push_destination(&mut destinations, RequestTraceDestination::File);
            }
            "jsonl_gz" => {
                legacy_jsonl_gz = true;
                push_destination(&mut destinations, RequestTraceDestination::File);
            }
            other => tracing::warn!(%other, "request trace: unknown destination ignored"),
        }
    }

    let compression = if legacy_jsonl_gz {
        Some(RequestTraceFileCompression::Gzip)
    } else if legacy_jsonl {
        Some(RequestTraceFileCompression::None)
    } else {
        None
    };

    (destinations, compression)
}

fn push_destination(
    destinations: &mut Vec<RequestTraceDestination>,
    destination: RequestTraceDestination,
) {
    if !destinations.contains(&destination) {
        destinations.push(destination);
    }
}

fn env_trimmed(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_usize(primary: &str, legacy: &str) -> Option<usize> {
    std::env::var(primary)
        .or_else(|_| std::env::var(legacy))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

fn env_u64(primary: &str, legacy: &str) -> Option<u64> {
    std::env::var(primary)
        .or_else(|_| std::env::var(legacy))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
}

fn parse_file_format(value: &str) -> RequestTraceFileFormat {
    match value.trim().to_lowercase().as_str() {
        "jsonl" => RequestTraceFileFormat::Jsonl,
        other => {
            tracing::warn!(
                %other,
                "request trace: unknown file format ignored; defaulting to jsonl"
            );
            RequestTraceFileFormat::Jsonl
        }
    }
}

fn parse_file_compression(value: &str) -> RequestTraceFileCompression {
    match value.trim().to_lowercase().as_str() {
        "none" | "off" | "false" => RequestTraceFileCompression::None,
        "gzip" | "gz" | "jsonl_gz" => RequestTraceFileCompression::Gzip,
        other => {
            tracing::warn!(
                %other,
                "request trace: unknown file compression ignored; defaulting to gzip"
            );
            RequestTraceFileCompression::Gzip
        }
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

    use super::*;

    const ALL_ENV_NAMES: &[&str] = &[
        env_request_trace::DYN_REQUEST_TRACE,
        env_request_trace::DYN_REQUEST_TRACE_DESTINATIONS,
        env_request_trace::DYN_REQUEST_TRACE_SINKS,
        env_request_trace::DYN_REQUEST_TRACE_FILE_PATH,
        env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH,
        env_request_trace::DYN_REQUEST_TRACE_FILE_FORMAT,
        env_request_trace::DYN_REQUEST_TRACE_FILE_COMPRESSION,
        env_request_trace::DYN_REQUEST_TRACE_CAPACITY,
        env_request_trace::DYN_REQUEST_TRACE_FILE_BUFFER_BYTES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES,
        env_request_trace::DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS,
        env_request_trace::DYN_REQUEST_TRACE_FILE_ROLL_BYTES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES,
        env_request_trace::DYN_REQUEST_TRACE_FILE_ROLL_LINES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES,
        env_request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
        env_request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC,
    ];

    fn with_request_trace_env<F>(overrides: &[(&'static str, &'static str)], test: F)
    where
        F: FnOnce(),
    {
        let mut vars: Vec<(&'static str, Option<&'static str>)> = ALL_ENV_NAMES
            .iter()
            .copied()
            .map(|name| (name, None))
            .collect();
        for &(name, value) in overrides {
            if let Some((_, current)) = vars.iter_mut().find(|(env_name, _)| *env_name == name) {
                *current = Some(value);
            } else {
                vars.push((name, Some(value)));
            }
        }
        temp_env::with_vars(vars, test);
    }

    #[test]
    #[serial_test::serial]
    fn master_switch_enables_default_file_destination_and_path() {
        with_request_trace_env(&[(env_request_trace::DYN_REQUEST_TRACE, "1")], || {
            let policy = load_from_env();
            assert!(policy.enabled);
            assert_eq!(policy.destinations, vec![RequestTraceDestination::File]);
            assert_eq!(policy.file_path.as_deref(), Some(DEFAULT_FILE_PATH));
            assert_eq!(policy.file_format, RequestTraceFileFormat::Jsonl);
            assert_eq!(policy.file_compression, RequestTraceFileCompression::Gzip);
        });
    }

    #[test]
    #[serial_test::serial]
    fn master_switch_yields_to_new_destination_overrides() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE, "1"),
                (
                    env_request_trace::DYN_REQUEST_TRACE_DESTINATIONS,
                    "file,stderr",
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_FILE_PATH,
                    "/tmp/custom-request-trace",
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_FILE_COMPRESSION,
                    "none",
                ),
                (env_request_trace::DYN_REQUEST_TRACE_FILE_ROLL_LINES, "10"),
                (
                    env_request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
                    "tcp://127.0.0.1:9999",
                ),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(
                    policy.destinations,
                    vec![
                        RequestTraceDestination::File,
                        RequestTraceDestination::Stderr
                    ]
                );
                assert_eq!(
                    policy.file_path.as_deref(),
                    Some("/tmp/custom-request-trace")
                );
                assert_eq!(policy.file_compression, RequestTraceFileCompression::None);
                assert_eq!(policy.file_roll_lines, Some(10));
                assert_eq!(
                    policy.tool_events_zmq_endpoint.as_deref(),
                    Some("tcp://127.0.0.1:9999")
                );
                assert_eq!(
                    policy.tool_events_zmq_topic.as_deref(),
                    Some("agent-tool-events")
                );
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn legacy_jsonl_sink_maps_to_file_destination() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE, "1"),
                (env_request_trace::DYN_REQUEST_TRACE_SINKS, "jsonl,stderr"),
                (
                    env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH,
                    "/tmp/legacy-request-trace",
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS,
                    "25",
                ),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(
                    policy.destinations,
                    vec![
                        RequestTraceDestination::File,
                        RequestTraceDestination::Stderr
                    ]
                );
                assert_eq!(
                    policy.file_path.as_deref(),
                    Some("/tmp/legacy-request-trace")
                );
                assert_eq!(policy.file_compression, RequestTraceFileCompression::None);
                assert_eq!(policy.file_flush_interval_ms, 25);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn legacy_jsonl_gz_sink_maps_to_gzip_file_destination() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE, "1"),
                (env_request_trace::DYN_REQUEST_TRACE_SINKS, "jsonl_gz"),
                (
                    env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES,
                    "20",
                ),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(policy.destinations, vec![RequestTraceDestination::File]);
                assert_eq!(policy.file_path.as_deref(), Some(DEFAULT_FILE_PATH));
                assert_eq!(policy.file_compression, RequestTraceFileCompression::Gzip);
                assert_eq!(policy.file_roll_lines, Some(20));
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn tool_event_topic_override_requires_endpoint() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE, "1"),
                (
                    env_request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
                    "tcp://127.0.0.1:9999",
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC,
                    "custom-tool-events",
                ),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(
                    policy.tool_events_zmq_topic.as_deref(),
                    Some("custom-tool-events")
                );
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn disabled_by_default() {
        with_request_trace_env(&[], || {
            let policy = load_from_env();
            assert!(!policy.enabled);
            assert!(policy.destinations.is_empty());
            assert!(policy.file_path.is_none());
            assert!(policy.tool_events_zmq_endpoint.is_none());
            assert!(policy.tool_events_zmq_topic.is_none());
        });
    }
}
