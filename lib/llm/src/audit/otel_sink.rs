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

use std::time::SystemTime;

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::logging::otlp as env_otlp;
use opentelemetry::Context;
use opentelemetry::logs::{AnyValue, LogRecord, Logger, LoggerProvider, Severity};
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::logs::{SdkLogger, SdkLoggerProvider};

use super::config::AuditPolicy;
use super::handle::AuditRecord;
use super::sink::AuditSink;

const DEFAULT_OTLP_HTTP_LOGS_ENDPOINT: &str = "http://localhost:4318/v1/logs";
const DEFAULT_OTLP_GRPC_ENDPOINT: &str = "http://localhost:4317";

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
    provider: SdkLoggerProvider,
    logger: SdkLogger,
    max_payload_bytes: usize,
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
        // Accept only the two values the runtime parser accepts (`grpc`,
        // `http/protobuf`); anything else — including the `http`/`http/proto`
        // aliases — warns and falls back to grpc, so audit and runtime resolve
        // the same protocol for the same env.
        match raw.trim().to_ascii_lowercase().as_str() {
            "grpc" => Self::Grpc,
            "http/protobuf" => Self::HttpProtobuf,
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
    // `std::env::var` returns Ok("") for a set-but-empty var; treat empty as unset
    // and fall through, matching the runtime's resolve_otlp_endpoint.
    if let Some(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT)
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        return endpoint;
    }

    if let Some(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT)
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
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

impl OtelSink {
    fn new(provider: SdkLoggerProvider, max_payload_bytes: usize) -> Self {
        let logger = provider.logger(AUDIT_INSTRUMENTATION_SCOPE);
        Self {
            provider,
            logger,
            max_payload_bytes,
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

        Ok(Self::new(provider, policy.otel_max_payload_bytes))
    }

    /// Serialize an `AuditRecord` into the `payload` attribute string.
    ///
    /// Pure-CPU and the bulk of `OtelSink::emit`'s cost. Called on the audit
    /// worker task (the bus consumer), off the request hot path — see `emit`.
    fn payload_for_limit(
        rec: &AuditRecord,
        max_payload_bytes: usize,
    ) -> Option<(String, bool, Option<String>)> {
        let payload = match serde_json::to_string(rec) {
            Ok(s) => s,
            Err(err) => {
                tracing::warn!(target: "dynamo_llm::audit", error = %err, "audit otel: serialize failed");
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

    // The marker is an `AuditRecord` with the payload dropped, so it shares the
    // same schema as a normal record — consumers parse one shape either way.
    let marker = AuditRecord {
        schema_version: rec.schema_version,
        request_id: rec.request_id.clone(),
        requested_streaming: rec.requested_streaming,
        model: rec.model.clone(),
        event_time: rec.event_time,
        request: None,
        response: None,
        audit_complete: false,
        audit_drop_reason: Some(reason.clone()),
    };

    match serde_json::to_string(&marker) {
        Ok(s) => Some((s, false, Some(reason))),
        Err(err) => {
            tracing::warn!(target: "dynamo_llm::audit", error = %err, "audit otel: marker serialize failed");
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
        let payload_result = Self::payload_for_limit(rec, self.max_payload_bytes);
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
        record.set_body(AnyValue::String(AUDIT_ENDPOINT_CHAT_COMPLETION.into()));
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
    use serial_test::serial;
    use std::sync::Arc;

    fn sample_record() -> AuditRecord {
        AuditRecord {
            schema_version: 1,
            request_id: "req-otel-1".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            event_time: SystemTime::now(),
            request: None,
            response: None,
            audit_complete: true,
            audit_drop_reason: None,
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
            audit_complete: true,
            audit_drop_reason: None,
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
            audit_complete: true,
            audit_drop_reason: None,
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
        let (payload, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, usize::MAX).expect("payload serializes");
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
            let (payload, complete, drop_reason) =
                OtelSink::payload_for_limit(&rec, usize::MAX).expect("payload serializes");
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
    fn payload_over_limit_emits_incomplete_marker() {
        let rec = sample_record();
        let (payload, audit_complete, audit_drop_reason) =
            OtelSink::payload_for_limit(&rec, 1).unwrap();

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
        let exact = serde_json::to_string(&rec).expect("serializes").len();

        let (payload, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, exact).expect("fits at boundary");
        assert!(complete, "payload exactly at the limit must be complete");
        assert!(drop_reason.is_none());
        assert_eq!(payload.len(), exact);

        let (_marker, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, exact - 1).expect("marker serializes");
        assert!(!complete, "one byte over the limit must emit the marker");
        assert!(drop_reason.unwrap().starts_with("otel_payload_too_large:"));
    }

    #[test]
    fn over_limit_marker_preserves_record_identity() {
        // Oversized records must stay identifiable (not silently dropped): the
        // marker keeps schema_version / request_id / model / streaming.
        let rec = sample_record();
        let (payload, _complete, _reason) =
            OtelSink::payload_for_limit(&rec, 1).expect("marker serializes");

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
    fn protocol_env_http_alias_falls_back_to_grpc_matching_runtime() {
        // The runtime parser accepts only `grpc`/`http/protobuf`, so `http`
        // must fall back to grpc here too — otherwise audit and runtime would
        // resolve different transports/ports for the same env.
        temp_env::with_vars(
            [
                (env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL, Some("http")),
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
