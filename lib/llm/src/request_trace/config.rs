// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use dynamo_runtime::config::env_is_truthy;
use dynamo_runtime::config::environment_names::llm::audit as env_audit;
use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;

use crate::telemetry::parse_sink_names;

use super::DEFAULT_TOOL_EVENTS_TOPIC;

const DEFAULT_CAPACITY: usize = 1024;
const DEFAULT_FILE_BUFFER_BYTES: usize = 1024 * 1024;
const DEFAULT_FILE_FLUSH_INTERVAL_MS: u64 = 1000;
const DEFAULT_FILE_ROLL_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_FILE_PATH: &str = "/tmp/dynamo-request-trace";
const DEFAULT_NATS_SUBJECT: &str = "dynamo.request_trace.v1";
const DEFAULT_LEGACY_AUDIT_NATS_SUBJECT: &str = "dynamo.audit.v1";
const DEFAULT_OTEL_MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;

const CAPTURE_UNINITIALIZED: u8 = 0;
const CAPTURE_ACTIVE: u8 = 1;
const CAPTURE_INACTIVE: u8 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestTraceSinkKind {
    File,
    Stderr,
    Nats,
    Otel,
}

impl RequestTraceSinkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Stderr => "stderr",
            Self::Nats => "nats",
            Self::Otel => "otel",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestTraceRecordKind {
    RequestEnd,
    RequestPayload,
    Tool,
}

impl RequestTraceRecordKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RequestEnd => "request_end",
            Self::RequestPayload => "request_payload",
            Self::Tool => "tool",
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
    pub records: Vec<RequestTraceRecordKind>,
    pub sinks: Vec<RequestTraceSinkKind>,
    pub file_path: Option<String>,
    pub file_format: RequestTraceFileFormat,
    pub file_compression: RequestTraceFileCompression,
    pub capacity: usize,
    pub file_buffer_bytes: usize,
    pub file_flush_interval_ms: u64,
    pub file_roll_bytes: u64,
    pub file_roll_lines: Option<u64>,
    pub nats_subject: String,
    pub otel_max_payload_bytes: usize,
    pub tool_events_zmq_endpoint: Option<String>,
    pub tool_events_zmq_topic: Option<String>,
}

impl RequestTracePolicy {
    pub fn sink_names(&self) -> Vec<&'static str> {
        self.sinks.iter().map(|sink| sink.as_str()).collect()
    }

    pub fn record_names(&self) -> Vec<&'static str> {
        self.records.iter().map(|record| record.as_str()).collect()
    }

    pub fn emit_request_end_records(&self) -> bool {
        self.records.contains(&RequestTraceRecordKind::RequestEnd)
    }

    pub fn emit_request_payload_records(&self) -> bool {
        self.records
            .contains(&RequestTraceRecordKind::RequestPayload)
    }

    pub fn emit_tool_records(&self) -> bool {
        self.records.contains(&RequestTraceRecordKind::Tool)
    }
}

static POLICY: OnceLock<RequestTracePolicy> = OnceLock::new();
static CAPTURE_STATE: AtomicU8 = AtomicU8::new(CAPTURE_UNINITIALIZED);

fn load_from_env() -> RequestTracePolicy {
    let legacy_audit_sinks = env_trimmed(env_audit::DYN_AUDIT_SINKS);
    let request_trace_enabled = env_is_truthy(env_request_trace::DYN_REQUEST_TRACE);
    let audit_force_logging = env_is_truthy(env_audit::DYN_AUDIT_FORCE_LOGGING);
    let records = load_records(request_trace_enabled, audit_force_logging);
    let enabled = !records.is_empty();
    let (sinks, legacy_file_compression, legacy_audit_sinks_selected) =
        load_sinks(enabled, legacy_audit_sinks.as_deref());
    let has_file_sink = sinks.contains(&RequestTraceSinkKind::File);
    let file_path = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_FILE_PATH)
        .or_else(|| env_trimmed(env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH))
        .or_else(|| env_trimmed(env_audit::DYN_AUDIT_OUTPUT_PATH))
        .or_else(|| (enabled && has_file_sink).then(|| DEFAULT_FILE_PATH.to_string()));
    let file_format = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_FILE_FORMAT)
        .as_deref()
        .map(parse_file_format)
        .unwrap_or(RequestTraceFileFormat::Jsonl);
    let file_compression = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_FILE_COMPRESSION)
        .as_deref()
        .map(parse_file_compression)
        .unwrap_or_else(|| legacy_file_compression.unwrap_or(RequestTraceFileCompression::Gzip));
    let capacity = env_usize(&[
        env_request_trace::DYN_REQUEST_TRACE_CAPACITY,
        env_audit::DYN_AUDIT_CAPACITY,
    ])
    .filter(|value| *value > 0)
    .unwrap_or(DEFAULT_CAPACITY);
    let file_buffer_bytes = env_usize(&[
        env_request_trace::DYN_REQUEST_TRACE_FILE_BUFFER_BYTES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES,
        env_audit::DYN_AUDIT_JSONL_BUFFER_BYTES,
    ])
    .unwrap_or(DEFAULT_FILE_BUFFER_BYTES);
    let file_flush_interval_ms = env_u64(&[
        env_request_trace::DYN_REQUEST_TRACE_FILE_FLUSH_INTERVAL_MS,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS,
        env_audit::DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS,
    ])
    .unwrap_or(DEFAULT_FILE_FLUSH_INTERVAL_MS);
    let file_roll_bytes = env_u64(&[
        env_request_trace::DYN_REQUEST_TRACE_FILE_ROLL_BYTES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES,
        env_audit::DYN_AUDIT_JSONL_GZ_ROLL_BYTES,
    ])
    .filter(|value| *value > 0)
    .unwrap_or(DEFAULT_FILE_ROLL_BYTES);
    let file_roll_lines = env_u64(&[
        env_request_trace::DYN_REQUEST_TRACE_FILE_ROLL_LINES,
        env_request_trace::DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES,
        env_audit::DYN_AUDIT_JSONL_GZ_ROLL_LINES,
    ])
    .filter(|value| *value > 0);
    let nats_subject = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_NATS_SUBJECT)
        .or_else(|| env_trimmed(env_audit::DYN_AUDIT_NATS_SUBJECT))
        .unwrap_or_else(|| {
            if legacy_audit_sinks_selected && sinks.contains(&RequestTraceSinkKind::Nats) {
                DEFAULT_LEGACY_AUDIT_NATS_SUBJECT.to_string()
            } else {
                DEFAULT_NATS_SUBJECT.to_string()
            }
        });
    let otel_max_payload_bytes = env_usize(&[
        env_request_trace::DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES,
        env_audit::DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES,
    ])
    .filter(|value| *value > 0)
    .unwrap_or(DEFAULT_OTEL_MAX_PAYLOAD_BYTES);
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
        records,
        sinks,
        file_path,
        file_format,
        file_compression,
        capacity,
        file_buffer_bytes,
        file_flush_interval_ms,
        file_roll_bytes,
        file_roll_lines,
        nats_subject,
        otel_max_payload_bytes,
        tool_events_zmq_endpoint,
        tool_events_zmq_topic,
    }
}

fn load_records(
    request_trace_enabled: bool,
    audit_force_logging: bool,
) -> Vec<RequestTraceRecordKind> {
    if let Some(value) = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_RECORDS) {
        return parse_record_kind_names(&value);
    }

    let mut records = Vec::new();
    if request_trace_enabled {
        push_record(&mut records, RequestTraceRecordKind::RequestEnd);
        push_record(&mut records, RequestTraceRecordKind::Tool);
    }
    if audit_force_logging {
        push_record(&mut records, RequestTraceRecordKind::RequestPayload);
    }
    records
}

fn parse_record_kind_names(value: &str) -> Vec<RequestTraceRecordKind> {
    let mut records = Vec::new();
    for name in value
        .split(',')
        .map(|v| v.trim().to_lowercase())
        .filter(|v| !v.is_empty())
    {
        match name.as_str() {
            "request_end" => push_record(&mut records, RequestTraceRecordKind::RequestEnd),
            "request_payload" => push_record(&mut records, RequestTraceRecordKind::RequestPayload),
            "tool" | "tools" | "tool_start" | "tool_end" | "tool_error" => {
                push_record(&mut records, RequestTraceRecordKind::Tool)
            }
            other => tracing::warn!(%other, "request trace: unknown record type ignored"),
        }
    }
    records
}

fn push_record(records: &mut Vec<RequestTraceRecordKind>, record: RequestTraceRecordKind) {
    if !records.contains(&record) {
        records.push(record);
    }
}

fn load_sinks(
    enabled: bool,
    legacy_audit_sinks: Option<&str>,
) -> (
    Vec<RequestTraceSinkKind>,
    Option<RequestTraceFileCompression>,
    bool,
) {
    if !enabled {
        return (Vec::new(), None, false);
    }

    if let Some(value) = env_trimmed(env_request_trace::DYN_REQUEST_TRACE_SINKS) {
        let (sinks, compression) = parse_sink_kind_names(&value);
        (sinks, compression, false)
    } else if let Some(value) = legacy_audit_sinks {
        let (sinks, compression) = parse_sink_kind_names(value);
        (sinks, compression, true)
    } else {
        (vec![RequestTraceSinkKind::File], None, false)
    }
}

fn parse_sink_kind_names(
    value: &str,
) -> (
    Vec<RequestTraceSinkKind>,
    Option<RequestTraceFileCompression>,
) {
    let mut sinks = Vec::new();
    let mut legacy_jsonl = false;
    let mut legacy_jsonl_gz = false;

    for name in parse_sink_names(value) {
        match name.as_str() {
            "file" => push_sink(&mut sinks, RequestTraceSinkKind::File),
            "stderr" => push_sink(&mut sinks, RequestTraceSinkKind::Stderr),
            "nats" => push_sink(&mut sinks, RequestTraceSinkKind::Nats),
            "otel" => push_sink(&mut sinks, RequestTraceSinkKind::Otel),
            "jsonl" => {
                legacy_jsonl = true;
                push_sink(&mut sinks, RequestTraceSinkKind::File);
            }
            "jsonl_gz" => {
                legacy_jsonl_gz = true;
                push_sink(&mut sinks, RequestTraceSinkKind::File);
            }
            other => tracing::warn!(%other, "request trace: unknown sink ignored"),
        }
    }

    let compression = if legacy_jsonl_gz {
        Some(RequestTraceFileCompression::Gzip)
    } else if legacy_jsonl {
        Some(RequestTraceFileCompression::None)
    } else {
        None
    };

    (sinks, compression)
}

fn push_sink(sinks: &mut Vec<RequestTraceSinkKind>, sink: RequestTraceSinkKind) {
    if !sinks.contains(&sink) {
        sinks.push(sink);
    }
}

fn env_trimmed(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_usize(names: &[&str]) -> Option<usize> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    })
}

fn env_u64(names: &[&str]) -> Option<u64> {
    names.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
    })
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

pub(crate) fn mark_capture_active() {
    CAPTURE_STATE.store(CAPTURE_ACTIVE, Ordering::Release);
}

pub(crate) fn mark_capture_inactive() {
    CAPTURE_STATE.store(CAPTURE_INACTIVE, Ordering::Release);
}

pub(crate) fn capture_enabled() -> bool {
    let policy = policy();
    policy.enabled && CAPTURE_STATE.load(Ordering::Acquire) == CAPTURE_ACTIVE
}

#[cfg(test)]
mod tests {
    use dynamo_runtime::config::environment_names::llm::audit as env_audit;
    use dynamo_runtime::config::environment_names::llm::request_trace as env_request_trace;

    use super::*;

    const ALL_ENV_NAMES: &[&str] = &[
        env_request_trace::DYN_REQUEST_TRACE,
        env_request_trace::DYN_REQUEST_TRACE_SINKS,
        env_request_trace::DYN_REQUEST_TRACE_FILE_PATH,
        env_request_trace::DYN_REQUEST_TRACE_OUTPUT_PATH,
        env_request_trace::DYN_REQUEST_TRACE_FILE_FORMAT,
        env_request_trace::DYN_REQUEST_TRACE_FILE_COMPRESSION,
        env_request_trace::DYN_REQUEST_TRACE_CAPACITY,
        env_request_trace::DYN_REQUEST_TRACE_RECORDS,
        env_request_trace::DYN_REQUEST_TRACE_NATS_SUBJECT,
        env_request_trace::DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES,
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
        env_audit::DYN_AUDIT_SINKS,
        env_audit::DYN_AUDIT_FORCE_LOGGING,
        env_audit::DYN_AUDIT_CAPACITY,
        env_audit::DYN_AUDIT_NATS_SUBJECT,
        env_audit::DYN_AUDIT_OUTPUT_PATH,
        env_audit::DYN_AUDIT_JSONL_BUFFER_BYTES,
        env_audit::DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS,
        env_audit::DYN_AUDIT_JSONL_GZ_ROLL_BYTES,
        env_audit::DYN_AUDIT_JSONL_GZ_ROLL_LINES,
        env_audit::DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES,
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
    fn master_switch_enables_default_request_end_and_tool_records() {
        with_request_trace_env(&[(env_request_trace::DYN_REQUEST_TRACE, "1")], || {
            let policy = load_from_env();
            assert!(policy.enabled);
            assert_eq!(
                policy.records,
                vec![
                    RequestTraceRecordKind::RequestEnd,
                    RequestTraceRecordKind::Tool
                ]
            );
            assert!(policy.emit_request_end_records());
            assert!(!policy.emit_request_payload_records());
            assert!(policy.emit_tool_records());
            assert_eq!(policy.sinks, vec![RequestTraceSinkKind::File]);
            assert_eq!(policy.file_path.as_deref(), Some(DEFAULT_FILE_PATH));
            assert_eq!(policy.file_format, RequestTraceFileFormat::Jsonl);
            assert_eq!(policy.file_compression, RequestTraceFileCompression::Gzip);
            assert_eq!(policy.nats_subject, DEFAULT_NATS_SUBJECT);
            assert_eq!(
                policy.otel_max_payload_bytes,
                DEFAULT_OTEL_MAX_PAYLOAD_BYTES
            );
        });
    }

    #[test]
    #[serial_test::serial]
    fn explicit_record_selector_enables_payload_without_request_end() {
        with_request_trace_env(
            &[(
                env_request_trace::DYN_REQUEST_TRACE_RECORDS,
                " request_payload, tool_start, request_payload, unknown ",
            )],
            || {
                let policy = load_from_env();
                assert!(policy.enabled);
                assert_eq!(
                    policy.records,
                    vec![
                        RequestTraceRecordKind::RequestPayload,
                        RequestTraceRecordKind::Tool,
                    ]
                );
                assert!(!policy.emit_request_end_records());
                assert!(policy.emit_request_payload_records());
                assert!(policy.emit_tool_records());
                assert_eq!(policy.sinks, vec![RequestTraceSinkKind::File]);
                assert_eq!(policy.file_path.as_deref(), Some(DEFAULT_FILE_PATH));
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn empty_record_selector_disables_trace_even_with_sink() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE_RECORDS, " , "),
                (env_request_trace::DYN_REQUEST_TRACE_SINKS, "nats"),
            ],
            || {
                let policy = load_from_env();
                assert!(!policy.enabled);
                assert!(policy.records.is_empty());
                assert!(policy.sinks.is_empty());
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn master_switch_yields_to_new_sink_overrides() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE, "1"),
                (
                    env_request_trace::DYN_REQUEST_TRACE_RECORDS,
                    "request_end,request_payload,tool",
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_SINKS,
                    "file,stderr,nats,otel",
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
                    env_request_trace::DYN_REQUEST_TRACE_NATS_SUBJECT,
                    "custom.request.trace",
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_OTEL_MAX_PAYLOAD_BYTES,
                    "1234",
                ),
                (
                    env_request_trace::DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT,
                    "tcp://127.0.0.1:9999",
                ),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(
                    policy.records,
                    vec![
                        RequestTraceRecordKind::RequestEnd,
                        RequestTraceRecordKind::RequestPayload,
                        RequestTraceRecordKind::Tool,
                    ]
                );
                assert_eq!(
                    policy.sinks,
                    vec![
                        RequestTraceSinkKind::File,
                        RequestTraceSinkKind::Stderr,
                        RequestTraceSinkKind::Nats,
                        RequestTraceSinkKind::Otel,
                    ]
                );
                assert_eq!(
                    policy.file_path.as_deref(),
                    Some("/tmp/custom-request-trace")
                );
                assert_eq!(policy.file_compression, RequestTraceFileCompression::None);
                assert_eq!(policy.file_roll_lines, Some(10));
                assert_eq!(policy.nats_subject, "custom.request.trace");
                assert_eq!(policy.otel_max_payload_bytes, 1234);
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
    fn legacy_jsonl_sink_maps_to_file_sink() {
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
                    policy.sinks,
                    vec![RequestTraceSinkKind::File, RequestTraceSinkKind::Stderr]
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
    fn legacy_jsonl_gz_sink_maps_to_gzip_file_sink() {
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
                assert_eq!(policy.sinks, vec![RequestTraceSinkKind::File]);
                assert_eq!(policy.file_path.as_deref(), Some(DEFAULT_FILE_PATH));
                assert_eq!(policy.file_compression, RequestTraceFileCompression::Gzip);
                assert_eq!(policy.file_roll_lines, Some(20));
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn legacy_audit_sinks_do_not_enable_request_trace_records() {
        with_request_trace_env(
            &[
                (env_audit::DYN_AUDIT_SINKS, "nats,otel"),
                (env_audit::DYN_AUDIT_NATS_SUBJECT, "legacy.audit.subject"),
                (env_audit::DYN_AUDIT_OTEL_MAX_PAYLOAD_BYTES, "5678"),
            ],
            || {
                let policy = load_from_env();
                assert!(!policy.enabled);
                assert!(policy.records.is_empty());
                assert!(policy.sinks.is_empty());
                assert_eq!(policy.nats_subject, "legacy.audit.subject");
                assert_eq!(policy.otel_max_payload_bytes, 5678);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn legacy_audit_force_logging_uses_truthy_values() {
        for value in ["1", "yes", "on", "true"] {
            with_request_trace_env(&[(env_audit::DYN_AUDIT_FORCE_LOGGING, value)], || {
                let policy = load_from_env();
                assert_eq!(policy.records, vec![RequestTraceRecordKind::RequestPayload]);
            });
        }
    }

    #[test]
    #[serial_test::serial]
    fn legacy_audit_force_logging_adds_payload_to_request_trace_defaults() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE, "1"),
                (env_audit::DYN_AUDIT_FORCE_LOGGING, "true"),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(
                    policy.records,
                    vec![
                        RequestTraceRecordKind::RequestEnd,
                        RequestTraceRecordKind::Tool,
                        RequestTraceRecordKind::RequestPayload,
                    ]
                );
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn legacy_audit_force_logging_enables_request_payload_records() {
        with_request_trace_env(&[(env_audit::DYN_AUDIT_FORCE_LOGGING, "true")], || {
            let policy = load_from_env();
            assert!(policy.enabled);
            assert_eq!(policy.records, vec![RequestTraceRecordKind::RequestPayload]);
            assert!(!policy.emit_request_end_records());
            assert!(policy.emit_request_payload_records());
            assert!(!policy.emit_tool_records());
            assert_eq!(policy.sinks, vec![RequestTraceSinkKind::File]);
            assert_eq!(policy.file_path.as_deref(), Some(DEFAULT_FILE_PATH));
        });
    }

    #[test]
    #[serial_test::serial]
    fn legacy_audit_nats_sink_uses_legacy_default_subject_when_payloads_enabled() {
        with_request_trace_env(
            &[
                (env_audit::DYN_AUDIT_FORCE_LOGGING, "true"),
                (env_audit::DYN_AUDIT_SINKS, "nats"),
            ],
            || {
                let policy = load_from_env();
                assert!(policy.enabled);
                assert_eq!(policy.records, vec![RequestTraceRecordKind::RequestPayload]);
                assert_eq!(policy.sinks, vec![RequestTraceSinkKind::Nats]);
                assert_eq!(policy.nats_subject, DEFAULT_LEGACY_AUDIT_NATS_SUBJECT);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn request_trace_nats_sink_uses_request_trace_default_subject() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE, "1"),
                (env_request_trace::DYN_REQUEST_TRACE_SINKS, "nats"),
                (env_audit::DYN_AUDIT_SINKS, "nats"),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(policy.sinks, vec![RequestTraceSinkKind::Nats]);
                assert_eq!(policy.nats_subject, DEFAULT_NATS_SUBJECT);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn legacy_audit_jsonl_aliases_map_to_request_trace_file_config() {
        with_request_trace_env(
            &[
                (env_audit::DYN_AUDIT_SINKS, "jsonl_gz"),
                (env_audit::DYN_AUDIT_OUTPUT_PATH, "/tmp/legacy-audit"),
                (env_audit::DYN_AUDIT_FORCE_LOGGING, "true"),
                (env_audit::DYN_AUDIT_JSONL_BUFFER_BYTES, "64"),
                (env_audit::DYN_AUDIT_JSONL_FLUSH_INTERVAL_MS, "5"),
                (env_audit::DYN_AUDIT_JSONL_GZ_ROLL_BYTES, "100"),
                (env_audit::DYN_AUDIT_JSONL_GZ_ROLL_LINES, "2"),
            ],
            || {
                let policy = load_from_env();
                assert!(policy.enabled);
                assert_eq!(policy.records, vec![RequestTraceRecordKind::RequestPayload]);
                assert_eq!(policy.sinks, vec![RequestTraceSinkKind::File]);
                assert_eq!(policy.file_path.as_deref(), Some("/tmp/legacy-audit"));
                assert_eq!(policy.file_compression, RequestTraceFileCompression::Gzip);
                assert_eq!(policy.file_buffer_bytes, 64);
                assert_eq!(policy.file_flush_interval_ms, 5);
                assert_eq!(policy.file_roll_bytes, 100);
                assert_eq!(policy.file_roll_lines, Some(2));
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn request_trace_sinks_win_over_legacy_audit_sinks() {
        with_request_trace_env(
            &[
                (env_request_trace::DYN_REQUEST_TRACE, "1"),
                (env_request_trace::DYN_REQUEST_TRACE_SINKS, "stderr"),
                (env_audit::DYN_AUDIT_SINKS, "nats,otel,jsonl_gz"),
            ],
            || {
                let policy = load_from_env();
                assert_eq!(policy.sinks, vec![RequestTraceSinkKind::Stderr]);
                assert_eq!(
                    policy.records,
                    vec![
                        RequestTraceRecordKind::RequestEnd,
                        RequestTraceRecordKind::Tool
                    ]
                );
                assert!(policy.file_path.is_none());
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
            assert!(policy.records.is_empty());
            assert!(policy.sinks.is_empty());
            assert!(policy.file_path.is_none());
            assert!(policy.tool_events_zmq_endpoint.is_none());
            assert!(policy.tool_events_zmq_topic.is_none());
        });
    }
}
