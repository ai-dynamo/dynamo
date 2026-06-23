// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP exporter sink for the audit bus.
//!
//! Emits exactly one OTLP `LogRecord` per `AuditRecord`. The exporter is
//! constructed once at sink init (not per emit). Network I/O happens on the
//! SDK's internal batch processor; `emit()` is non-blocking enqueue. The audit
//! worker calls `force_flush()` after draining on shutdown, but abrupt process
//! teardown can still lose buffered OTLP records.
//!
//! Transport follows `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` with
//! `OTEL_EXPORTER_OTLP_PROTOCOL` as fallback. Supported values are `grpc`
//! (default) and `http/protobuf`. The default matches the runtime OTLP
//! exporter (`lib/runtime/src/logging.rs`), so audit logs and application
//! telemetry resolve the same protocol/endpoint from the shared env vars.

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
    time::SystemTime,
};

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use axum::http::HeaderValue;
use dynamo_runtime::config::environment_names::{
    llm::audit as env_audit, logging::otlp as env_otlp,
};
use opentelemetry::Context;
use opentelemetry::logs::{AnyValue, LogRecord, Logger, LoggerProvider, Severity};
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::logs::{SdkLogger, SdkLoggerProvider};
use serde_json::json;

use super::config::AuditPolicy;
use super::handle::{AuditHttpRequestHeaders, AuditRecord};
use super::sink::AuditSink;

const DEFAULT_OTLP_HTTP_LOGS_ENDPOINT: &str = "http://localhost:4318/v1/logs";
const DEFAULT_OTLP_GRPC_ENDPOINT: &str = "http://localhost:4317";
const DEFAULT_HTTP_HEADER_VALUE_MAX_BYTES: usize = 4 * 1024;
const DEFAULT_HTTP_HEADERS_MAX_BYTES: usize = 64 * 1024;
const REDACTED_HEADER_VALUE: &str = "[REDACTED]";

const DEFAULT_REDACTED_HTTP_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "api-key",
    "apikey",
    "x-api-key",
    "openai-api-key",
    "x-openai-api-key",
    "x-auth-token",
    "x-access-token",
    "x-csrf-token",
    "x-xsrf-token",
    "x-amz-security-token",
];

const SENSITIVE_HTTP_HEADER_NAME_FRAGMENTS: &[&str] =
    &["token", "secret", "password", "credential"];
const DEFAULT_REDACTION_EXEMPT_HTTP_HEADER_PREFIXES: &[&str] = &["nvcf-"];

/// Logical endpoint label so phase 2 (completions / responses) can be
/// distinguished without changing the body.
const AUDIT_ENDPOINT_CHAT_COMPLETION: &str = "openai.chat_completion";

/// Instrumentation scope name on the emitted `LogRecord`.
const AUDIT_INSTRUMENTATION_SCOPE: &str = "dynamo.payload";

/// Default service name when `OTEL_SERVICE_NAME` is unset.
const DEFAULT_SERVICE_NAME: &str = "dynamo";

pub struct OtelSink {
    /// Held so the SDK's batch processor stays alive for the sink's lifetime
    /// and can be force-flushed when the audit worker shuts down.
    #[allow(dead_code)]
    provider: SdkLoggerProvider,
    logger: SdkLogger,
    max_payload_bytes: usize,
    header_policy: Arc<OtelHeaderPolicy>,
}

#[derive(Clone, Debug)]
struct OtelHeaderPolicy {
    redact_names: HashSet<String>,
    max_value_bytes: usize,
    max_total_bytes: usize,
}

impl Default for OtelHeaderPolicy {
    fn default() -> Self {
        Self {
            redact_names: DEFAULT_REDACTED_HTTP_HEADERS
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            max_value_bytes: DEFAULT_HTTP_HEADER_VALUE_MAX_BYTES,
            max_total_bytes: DEFAULT_HTTP_HEADERS_MAX_BYTES,
        }
    }
}

impl OtelHeaderPolicy {
    fn from_env() -> Arc<Self> {
        let mut policy = Self::default();
        if let Ok(raw) = std::env::var(env_audit::DYN_AUDIT_OTEL_HTTP_HEADER_REDACT_LIST) {
            for name in raw.split(|c: char| c == ',' || c.is_whitespace()) {
                let name = name.trim().to_ascii_lowercase();
                if !name.is_empty() {
                    policy.redact_names.insert(name);
                }
            }
        }
        Arc::new(policy)
    }

    fn should_redact(&self, name: &str) -> bool {
        self.redact_names.contains(name)
            || (!DEFAULT_REDACTION_EXEMPT_HTTP_HEADER_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix))
                && SENSITIVE_HTTP_HEADER_NAME_FRAGMENTS
                    .iter()
                    .any(|fragment| name.contains(fragment)))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OtlpLogsProtocol {
    HttpProtobuf,
    Grpc,
}

impl OtlpLogsProtocol {
    fn from_env() -> Self {
        // Default to grpc when unset, matching the runtime OTLP exporter
        // (`parse_otlp_protocol` in lib/runtime/src/logging.rs). Diverging here
        // would silently send audit logs to a different protocol/port than the
        // rest of Dynamo's telemetry when only a generic endpoint is configured.
        let raw = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL)
            .or_else(|_| std::env::var(env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL))
            .unwrap_or_else(|_| "grpc".to_string());
        match raw.trim().to_ascii_lowercase().as_str() {
            "grpc" => Self::Grpc,
            "http/protobuf" | "http/proto" | "http" => Self::HttpProtobuf,
            other => {
                tracing::warn!(
                    protocol = other,
                    "audit otel: unsupported OTLP logs protocol; defaulting to grpc"
                );
                Self::Grpc
            }
        }
    }

    fn default_endpoint(self) -> &'static str {
        match self {
            Self::HttpProtobuf => DEFAULT_OTLP_HTTP_LOGS_ENDPOINT,
            Self::Grpc => DEFAULT_OTLP_GRPC_ENDPOINT,
        }
    }
}

fn logs_endpoint_from_env(protocol: OtlpLogsProtocol) -> String {
    if let Ok(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT) {
        return endpoint;
    }

    if let Ok(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT) {
        return match protocol {
            OtlpLogsProtocol::HttpProtobuf => {
                let trimmed = endpoint.trim_end_matches('/');
                format!("{trimmed}/v1/logs")
            }
            OtlpLogsProtocol::Grpc => endpoint,
        };
    }

    protocol.default_endpoint().to_string()
}

fn render_header_value(name: &str, value: &HeaderValue, policy: &OtelHeaderPolicy) -> String {
    if policy.should_redact(name) {
        return REDACTED_HEADER_VALUE.to_string();
    }

    let raw = value.as_bytes();
    if raw.len() > policy.max_value_bytes {
        return format!("[TRUNCATED: bytes={}]", raw.len());
    }

    match value.to_str() {
        Ok(value) => value.to_string(),
        Err(_) => format!("[NON_UTF8: bytes={}]", raw.len()),
    }
}

fn otel_http_request_value(
    headers: &AuditHttpRequestHeaders,
    policy: &OtelHeaderPolicy,
) -> serde_json::Value {
    let mut out = BTreeMap::new();
    let mut bytes_used = 0usize;
    let mut truncated = false;
    let header_map = headers.headers();

    'headers: for name in header_map.keys() {
        let name = name.as_str().to_ascii_lowercase();
        let mut values = Vec::new();

        for value in header_map.get_all(name.as_str()).iter() {
            let rendered = render_header_value(&name, value, policy);
            let next_bytes = name.len().saturating_add(rendered.len());
            if bytes_used.saturating_add(next_bytes) > policy.max_total_bytes {
                truncated = true;
                break 'headers;
            }
            bytes_used = bytes_used.saturating_add(next_bytes);
            values.push(serde_json::Value::String(rendered));
        }

        if values.len() == 1 {
            out.insert(name, values.remove(0));
        } else if !values.is_empty() {
            out.insert(name, serde_json::Value::Array(values));
        }
    }

    json!({
        "headers": out,
        "headers_truncated": truncated,
    })
}

impl OtelSink {
    fn new(
        provider: SdkLoggerProvider,
        max_payload_bytes: usize,
        header_policy: Arc<OtelHeaderPolicy>,
    ) -> Self {
        let logger = provider.logger(AUDIT_INSTRUMENTATION_SCOPE);
        Self {
            provider,
            logger,
            max_payload_bytes,
            header_policy,
        }
    }

    pub async fn from_policy(policy: &AuditPolicy) -> Result<Self> {
        let protocol = OtlpLogsProtocol::from_env();
        let endpoint = logs_endpoint_from_env(protocol);

        let exporter = match protocol {
            OtlpLogsProtocol::HttpProtobuf => opentelemetry_otlp::LogExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_endpoint(endpoint.clone())
                .build(),
            OtlpLogsProtocol::Grpc => opentelemetry_otlp::LogExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint.clone())
                .build(),
        }
        .with_context(|| {
            format!("building OTLP audit log exporter for endpoint {endpoint} using {protocol:?}")
        })?;

        let service_name = std::env::var(env_otlp::OTEL_SERVICE_NAME)
            .unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());
        let resource = Resource::builder_empty()
            .with_service_name(service_name)
            .build();

        let provider = SdkLoggerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource)
            .build();

        let header_policy = OtelHeaderPolicy::from_env();

        Ok(Self::new(
            provider,
            policy.otel_max_payload_bytes,
            header_policy,
        ))
    }

    fn serialize_payload(
        rec: &AuditRecord,
        header_policy: &OtelHeaderPolicy,
    ) -> serde_json::Result<String> {
        if rec.otel_http_headers.is_none() {
            return serde_json::to_string(rec);
        }

        let mut payload = serde_json::to_value(rec)?;
        if let Some(obj) = payload.as_object_mut()
            && let Some(headers) = &rec.otel_http_headers
        {
            obj.insert(
                "http".to_string(),
                json!({
                    "request": otel_http_request_value(headers, header_policy),
                }),
            );
        }
        serde_json::to_string(&payload)
    }

    /// Serialize an `AuditRecord` into the `payload` attribute string.
    ///
    /// Pure-CPU and the bulk of `OtelSink::emit`'s cost. Called on the audit
    /// worker task (the bus consumer), off the request hot path — see `emit`.
    fn payload_for_limit(
        rec: &AuditRecord,
        max_payload_bytes: usize,
        header_policy: &OtelHeaderPolicy,
    ) -> Option<(String, bool, Option<String>)> {
        let payload = match Self::serialize_payload(rec, header_policy) {
            Ok(s) => s,
            Err(err) => {
                tracing::warn!(target: "dynamo_llm::audit", "audit otel: serialize failed: {err}");
                return None;
            }
        };
        if payload.len() <= max_payload_bytes {
            return Some((payload, true, None));
        }

        marker_payload(
            rec,
            format!(
                "otel_payload_too_large:max_bytes={}:actual_bytes={}",
                max_payload_bytes,
                payload.len()
            ),
        )
    }
}

fn marker_payload(rec: &AuditRecord, reason: String) -> Option<(String, bool, Option<String>)> {
    tracing::warn!(
        target: "dynamo_llm::audit",
        request_id = %rec.request_id,
        audit_drop_reason = %reason,
        "audit otel: emitting incomplete marker"
    );

    let payload = json!({
        "schema_version": rec.schema_version,
        "request_id": &rec.request_id,
        "requested_streaming": rec.requested_streaming,
        "model": &rec.model,
        "audit_complete": false,
        "audit_drop_reason": reason,
    });

    match serde_json::to_string(&payload) {
        Ok(s) => Some((s, false, Some(reason))),
        Err(err) => {
            tracing::warn!(target: "dynamo_llm::audit", "audit otel: marker serialize failed: {err}");
            None
        }
    }
}

#[async_trait]
impl AuditSink for OtelSink {
    fn name(&self) -> &'static str {
        "otel"
    }

    async fn emit(&self, rec: &AuditRecord) {
        // OTLP Timestamp = when the event actually occurred (captured on the
        // producing thread at request arrival); set below from `rec.event_time`.
        // ObservedTimestamp = now, when this sink drained the record off the
        // audit bus. The gap between them is bus + sink-task latency, which is
        // exactly what we no longer want folded into Timestamp.
        let observed_timestamp = SystemTime::now();

        // Serialize the payload on the audit worker task. This runs on the bus
        // consumer, which is independent of the request future (inference has
        // already returned to the client by the time we drain the record), so
        // it does not block the request hot path. The OTEL SDK emit below only
        // enqueues to the BatchLogProcessor.
        let start = std::time::Instant::now();
        let payload_result =
            Self::payload_for_limit(rec, self.max_payload_bytes, &self.header_policy);
        tracing::debug!(
            target: "dynamo.audit.otel.serde",
            request_id = %rec.request_id,
            elapsed_us = start.elapsed().as_micros() as u64,
            payload_len = payload_result.as_ref().map(|(p, _, _)| p.len()).unwrap_or(0),
            "OTEL audit payload serialized"
        );
        let Some((payload, audit_complete, audit_drop_reason)) = payload_result else {
            return;
        };

        let mut record = self.logger.create_log_record();
        record.set_timestamp(rec.event_time);
        record.set_observed_timestamp(observed_timestamp);
        record.set_severity_number(Severity::Info);
        record.set_severity_text("INFO");
        record.set_body(AnyValue::String("openai.chat_completion".into()));
        record.add_attribute("rid", AnyValue::String(rec.request_id.clone().into()));
        record.add_attribute(
            "endpoint",
            AnyValue::String(AUDIT_ENDPOINT_CHAT_COMPLETION.into()),
        );
        record.add_attribute("model", AnyValue::String(rec.model.clone().into()));
        record.add_attribute("streaming", AnyValue::Boolean(rec.requested_streaming));
        record.add_attribute("audit_complete", AnyValue::Boolean(audit_complete));
        if let Some(reason) = audit_drop_reason {
            record.add_attribute("audit_drop_reason", AnyValue::String(reason.into()));
        }
        record.add_attribute("payload", AnyValue::String(payload.into()));

        // Audit OTLP export is an explicit sink, not telemetry generated while
        // exporting telemetry. Use a fresh context so a globally suppressed
        // tracing bridge cannot cause the direct LogRecord emit to be skipped.
        let _guard = Context::new().attach();
        self.logger.emit(record);
    }

    async fn shutdown(&self) {
        if let Err(err) = self.provider.force_flush() {
            tracing::warn!(
                target: "dynamo_llm::audit",
                error = %err,
                "audit otel: force_flush failed during shutdown"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    };
    use axum::http::{HeaderMap, HeaderValue};
    use serial_test::serial;

    fn sample_record() -> AuditRecord {
        AuditRecord {
            schema_version: 1,
            request_id: "req-otel-1".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            event_time: SystemTime::now(),
            request: None,
            response: None,
            otel_http_headers: None,
        }
    }

    /// Sample record with a full request payload that exercises every wire
    /// type the serializer has to encode — strings, ints, bools, **floats**
    /// (the sampling params: temperature/top_p/frequency_penalty/presence_penalty
    /// plus vLLM-style top_k/min_p/repetition_penalty), arrays of objects
    /// (messages), and nested objects (tools / nvext). The point of this
    /// record is to cover the round-trip path that production actually uses
    /// via `OtelSink::payload_for_limit`.
    fn sample_record_with_request() -> AuditRecord {
        let request_json = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with a single word."},
            ],
            "stream": true,
            "store": true,
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.25,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "max_tokens": 64,
            "logprobs": true,
            "top_logprobs": 3,
            "stop": ["END"],
            "n": 1,
            "seed": 42,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("construct test request");
        AuditRecord {
            schema_version: 1,
            request_id: "req-otel-with-floats".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            event_time: SystemTime::now(),
            request: Some(Arc::new(request)),
            response: None,
            otel_http_headers: None,
        }
    }

    fn sample_record_with_response(requested_streaming: bool) -> AuditRecord {
        let response_json = serde_json::json!({
            "id": "chatcmpl-response-fields",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The weather is clear.",
                    "reasoning_content": "I should call get_weather before answering.",
                    "tool_calls": [{
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 11,
                "total_tokens": 20
            }
        });
        let response: NvCreateChatCompletionResponse =
            serde_json::from_value(response_json).expect("construct test response");
        AuditRecord {
            schema_version: 1,
            request_id: "req-otel-response-fields".to_string(),
            requested_streaming,
            model: "test-model".to_string(),
            event_time: SystemTime::now(),
            request: None,
            response: Some(Arc::new(response)),
            otel_http_headers: None,
        }
    }

    /// Exercises the production serialization path: `payload_for_limit`
    /// → string → `serde_json::from_str` (what downstream consumers use to
    /// parse the `payload` attribute). Validates semantic round-trip on a
    /// record that contains floats (sampling params) + nested arrays/objects
    /// (messages, tools) — i.e. the same wire shape as a real chat-completion
    /// request.
    #[test]
    fn payload_for_limit_round_trips_a_full_request() {
        let rec = sample_record_with_request();
        let header_policy = OtelHeaderPolicy::default();
        let (payload, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, usize::MAX, &header_policy)
                .expect("payload serializes");
        assert!(complete);
        assert!(drop_reason.is_none());

        let decoded: AuditRecord =
            serde_json::from_str(&payload).expect("payload string decodes back to AuditRecord");
        assert_eq!(decoded.request_id, rec.request_id);
        assert_eq!(decoded.requested_streaming, rec.requested_streaming);
        assert_eq!(decoded.model, rec.model);

        // Round-trip the record through the JSON Value form to compare
        // structurally — sidesteps any field-ordering differences and proves
        // semantic equivalence (which is the only contract downstream
        // consumers rely on).
        let rec_value = serde_json::to_value(&rec).expect("rec serializes via serde_json");
        let decoded_value =
            serde_json::to_value(&decoded).expect("decoded serializes via serde_json");
        assert_eq!(rec_value, decoded_value);

        let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
        let request = &value["request"];
        assert!(value.get("response").is_none());
        assert!(request.get("inner").is_none());
        assert_eq!(request["model"], "test-model");
        assert_eq!(request["stream"], true);
        assert_eq!(request["temperature"], 0.7);
        assert_eq!(request["top_p"], 0.95);
        assert_eq!(request["frequency_penalty"], 0.5);
        assert_eq!(request["presence_penalty"], 0.25);
        assert_eq!(request["top_k"], 40);
        assert_eq!(request["min_p"], 0.05);
        assert_eq!(request["repetition_penalty"], 1.1);
        assert_eq!(request["max_tokens"], 64);
        assert_eq!(request["logprobs"], true);
        assert_eq!(request["top_logprobs"], 3);
        assert_eq!(request["stop"][0], "END");
        assert_eq!(request["seed"], 42);
        assert_eq!(request["tool_choice"], "auto");
        assert_eq!(request["parallel_tool_calls"], true);
        assert_eq!(request["tools"][0]["type"], "function");
        assert_eq!(request["messages"][1]["role"], "user");
    }

    #[test]
    fn payload_for_limit_preserves_response_content_reasoning_and_tool_calls() {
        for requested_streaming in [false, true] {
            let rec = sample_record_with_response(requested_streaming);
            let header_policy = OtelHeaderPolicy::default();
            let (payload, complete, drop_reason) =
                OtelSink::payload_for_limit(&rec, usize::MAX, &header_policy)
                    .expect("payload serializes");
            assert!(complete);
            assert!(drop_reason.is_none());

            let decoded: serde_json::Value = serde_json::from_str(&payload).unwrap();
            assert_eq!(decoded["requested_streaming"], requested_streaming);
            assert!(decoded.get("request").is_none());

            let message = &decoded["response"]["choices"][0]["message"];
            assert_eq!(message["content"], "The weather is clear.");
            assert_eq!(
                message["reasoning_content"],
                "I should call get_weather before answering."
            );
            assert_eq!(message["tool_calls"][0]["id"], "call_weather");
            assert_eq!(message["tool_calls"][0]["type"], "function");
            assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
            assert_eq!(
                message["tool_calls"][0]["function"]["arguments"],
                "{\"city\":\"Tokyo\"}"
            );
            assert_eq!(decoded["response"]["usage"]["total_tokens"], 20);
        }
    }

    #[test]
    fn payload_for_limit_injects_redacted_http_headers_only_for_otel_payload() {
        let mut headers = HeaderMap::new();
        headers.insert("accept", HeaderValue::from_static("application/json"));
        headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
        headers.insert("x-forwarded-for", HeaderValue::from_static("203.0.113.7"));
        headers.insert(
            "x-custom-token-name",
            HeaderValue::from_static("also-secret"),
        );

        let mut rec = sample_record();
        rec.otel_http_headers = Some(Arc::new(AuditHttpRequestHeaders::new(Arc::new(headers))));

        let normal_sink_value = serde_json::to_value(&rec).expect("record serializes");
        assert!(normal_sink_value.get("http").is_none());
        assert!(normal_sink_value.get("otel_http_headers").is_none());

        let header_policy = OtelHeaderPolicy::default();
        let (payload, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, usize::MAX, &header_policy)
                .expect("payload serializes");
        assert!(complete);
        assert!(drop_reason.is_none());

        let decoded: serde_json::Value = serde_json::from_str(&payload).unwrap();
        let decoded_headers = &decoded["http"]["request"]["headers"];
        assert_eq!(decoded_headers["accept"], "application/json");
        assert_eq!(decoded_headers["authorization"], REDACTED_HEADER_VALUE);
        assert_eq!(decoded_headers["x-forwarded-for"], "203.0.113.7");
        assert_eq!(
            decoded_headers["x-custom-token-name"],
            REDACTED_HEADER_VALUE
        );
        assert_eq!(decoded["http"]["request"]["headers_truncated"], false);
    }

    #[test]
    fn payload_for_limit_omits_http_when_no_headers_captured() {
        // When header capture is inactive (otel_http_headers = None) the payload
        // must not carry an `http` object — non-otel sinks and header-disabled
        // requests never emit headers.
        let rec = sample_record_with_request();
        assert!(rec.otel_http_headers.is_none());

        let header_policy = OtelHeaderPolicy::default();
        let (payload, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, usize::MAX, &header_policy)
                .expect("payload serializes");
        assert!(complete);
        assert!(drop_reason.is_none());

        let decoded: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert!(decoded.get("http").is_none());
    }

    #[test]
    #[serial]
    fn custom_redact_list_redacts_named_headers() {
        temp_env::with_vars(
            [(
                env_audit::DYN_AUDIT_OTEL_HTTP_HEADER_REDACT_LIST,
                Some("x-title"),
            )],
            || {
                let policy = OtelHeaderPolicy::from_env();
                assert!(policy.should_redact("x-title"));
            },
        );
    }

    #[test]
    fn default_policy_preserves_nvcf_headers() {
        let policy = OtelHeaderPolicy::default();
        assert!(!policy.should_redact("nvcf-function-id"));
        assert!(!policy.should_redact("nvcf-function-name"));
        assert!(!policy.should_redact("nvcf-ncaid"));
        assert!(!policy.should_redact("nvcf-sub"));
        assert!(!policy.should_redact("nvcf-token-shaped-name"));
    }

    #[test]
    fn payload_over_limit_emits_incomplete_marker() {
        let rec = sample_record();
        let header_policy = OtelHeaderPolicy::default();
        let (payload, audit_complete, audit_drop_reason) =
            OtelSink::payload_for_limit(&rec, 1, &header_policy).unwrap();

        assert!(!audit_complete);
        assert!(
            audit_drop_reason
                .unwrap()
                .starts_with("otel_payload_too_large:")
        );
        let decoded: serde_json::Value = serde_json::from_str(&payload).unwrap();

        assert_eq!(decoded["audit_complete"], false);
        assert!(
            decoded["audit_drop_reason"]
                .as_str()
                .unwrap()
                .starts_with("otel_payload_too_large:")
        );
        assert!(decoded.get("request").is_none());
        assert!(decoded.get("response").is_none());
    }

    #[test]
    fn payload_size_guard_boundary_is_inclusive() {
        // Lock the `payload.len() <= max_payload_bytes` boundary: a payload that
        // exactly fits is emitted complete; one byte tighter forces the marker.
        let rec = sample_record_with_request();
        let header_policy = OtelHeaderPolicy::default();
        let exact = OtelSink::serialize_payload(&rec, &header_policy)
            .expect("serializes")
            .len();

        let (payload, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, exact, &header_policy).expect("fits at boundary");
        assert!(complete, "payload exactly at the limit must be complete");
        assert!(drop_reason.is_none());
        assert_eq!(payload.len(), exact);

        let (_marker, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, exact - 1, &header_policy)
                .expect("marker serializes");
        assert!(!complete, "one byte over the limit must emit the marker");
        assert!(drop_reason.unwrap().starts_with("otel_payload_too_large:"));
    }

    #[test]
    fn over_limit_marker_preserves_record_identity() {
        // Oversized records must stay identifiable (not silently dropped): the
        // marker keeps schema_version / request_id / model / streaming.
        let rec = sample_record();
        let header_policy = OtelHeaderPolicy::default();
        let (payload, _complete, _reason) =
            OtelSink::payload_for_limit(&rec, 1, &header_policy).expect("marker serializes");

        let decoded: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(decoded["schema_version"], rec.schema_version);
        assert_eq!(decoded["request_id"], rec.request_id);
        assert_eq!(decoded["model"], rec.model);
        assert_eq!(decoded["requested_streaming"], rec.requested_streaming);
    }

    #[test]
    #[serial]
    fn protocol_env_defaults_to_grpc() {
        temp_env::with_vars(
            [
                (env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL, None::<&str>),
                (env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL, None::<&str>),
            ],
            // Matches the runtime OTLP exporter default (grpc) so audit logs and
            // application telemetry agree on protocol/endpoint when unset.
            || assert_eq!(OtlpLogsProtocol::from_env(), OtlpLogsProtocol::Grpc),
        );
    }

    #[test]
    #[serial]
    fn protocol_env_unknown_falls_back_to_grpc() {
        temp_env::with_vars(
            [
                (
                    env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL,
                    Some("carrier-pigeon"),
                ),
                (env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL, None::<&str>),
            ],
            || assert_eq!(OtlpLogsProtocol::from_env(), OtlpLogsProtocol::Grpc),
        );
    }

    #[test]
    #[serial]
    fn logs_protocol_takes_precedence_over_global() {
        temp_env::with_vars(
            [
                (env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL, Some("grpc")),
                (env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL, Some("http/protobuf")),
            ],
            || assert_eq!(OtlpLogsProtocol::from_env(), OtlpLogsProtocol::Grpc),
        );
    }

    #[test]
    #[serial]
    fn logs_endpoint_uses_signal_specific_endpoint_first() {
        temp_env::with_vars(
            [
                (
                    env_otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT,
                    Some("http://collector:9999/custom/logs"),
                ),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT,
                    Some("http://collector:4318"),
                ),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
                    Some("http://collector:4317/v1/traces"),
                ),
            ],
            || {
                assert_eq!(
                    logs_endpoint_from_env(OtlpLogsProtocol::HttpProtobuf),
                    "http://collector:9999/custom/logs"
                );
            },
        );
    }

    #[test]
    #[serial]
    fn logs_endpoint_falls_back_to_generic_endpoint_not_traces_endpoint() {
        temp_env::with_vars(
            [
                (env_otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT, None::<&str>),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT,
                    Some("http://collector:4318"),
                ),
                (
                    env_otlp::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
                    Some("http://collector:4317/v1/traces"),
                ),
            ],
            || {
                assert_eq!(
                    logs_endpoint_from_env(OtlpLogsProtocol::HttpProtobuf),
                    "http://collector:4318/v1/logs"
                );
                assert_eq!(
                    logs_endpoint_from_env(OtlpLogsProtocol::Grpc),
                    "http://collector:4318"
                );
            },
        );
    }
}
