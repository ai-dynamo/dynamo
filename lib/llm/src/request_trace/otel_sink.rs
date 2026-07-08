// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP exporter sink for request trace records.

use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context as _;
use anyhow::Result;
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::logging::otlp as env_otlp;
use opentelemetry::Context;
use opentelemetry::logs::AnyValue;
use opentelemetry::logs::LogRecord;
use opentelemetry::logs::Logger;
use opentelemetry::logs::LoggerProvider;
use opentelemetry::logs::Severity;
use opentelemetry_otlp::Protocol;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::logs::SdkLogger;
use opentelemetry_sdk::logs::SdkLoggerProvider;

use super::RequestTraceEventType;
use super::RequestTraceRecord;
use super::RequestTraceSchema;
use super::config::RequestTracePolicy;
use super::sink::RequestTraceSink;

const DEFAULT_OTLP_HTTP_LOGS_ENDPOINT: &str = "http://localhost:4318/v1/logs";
const DEFAULT_OTLP_GRPC_ENDPOINT: &str = "http://localhost:4317";
const REQUEST_TRACE_INSTRUMENTATION_SCOPE: &str = "dynamo.request_trace";
const DEFAULT_SERVICE_NAME: &str = "dynamo";

pub struct OtelRequestTraceSink {
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
        let raw = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL)
            .or_else(|_| std::env::var(env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL))
            .unwrap_or_else(|_| "grpc".to_string());
        match raw.trim().to_ascii_lowercase().as_str() {
            "grpc" => Self::Grpc,
            "http/protobuf" => Self::HttpProtobuf,
            other => {
                tracing::warn!(
                    protocol = other,
                    "request trace otel: unsupported OTLP logs protocol; defaulting to grpc"
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
    if let Some(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT)
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return endpoint;
    }

    if let Some(endpoint) = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_ENDPOINT)
        .ok()
        .filter(|value| !value.trim().is_empty())
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

impl OtelRequestTraceSink {
    fn new(provider: SdkLoggerProvider, max_payload_bytes: usize) -> Self {
        let logger = provider.logger(REQUEST_TRACE_INSTRUMENTATION_SCOPE);
        Self {
            provider,
            logger,
            max_payload_bytes,
        }
    }

    pub async fn from_policy(policy: &RequestTracePolicy) -> Result<Self> {
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
            format!(
                "building OTLP request trace exporter for endpoint {endpoint} using {protocol:?}"
            )
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

    fn payload_for_limit(
        record: &RequestTraceRecord,
        max_payload_bytes: usize,
    ) -> Option<(String, bool, Option<String>)> {
        let payload = match serde_json::to_string(record) {
            Ok(payload) => payload,
            Err(error) => {
                tracing::warn!(
                    target: "dynamo_llm::request_trace",
                    error = %error,
                    "request trace otel: serialize failed"
                );
                return None;
            }
        };
        if payload.len() <= max_payload_bytes {
            return Some((payload, true, None));
        }

        marker_payload(
            record,
            format!(
                "otel_payload_too_large:max_bytes={}:actual_bytes={}",
                max_payload_bytes,
                payload.len()
            ),
        )
    }
}

fn marker_payload(
    record: &RequestTraceRecord,
    reason: String,
) -> Option<(String, bool, Option<String>)> {
    let Some(payload) = record.payload.as_ref() else {
        tracing::warn!(
            target: "dynamo_llm::request_trace",
            event_type = event_type_name(record.event_type),
            drop_reason = %reason,
            "request trace otel: dropping oversized non-payload record"
        );
        return None;
    };

    tracing::warn!(
        target: "dynamo_llm::request_trace",
        request_id = %payload.request_id,
        payload_drop_reason = %reason,
        "request trace otel: emitting incomplete marker"
    );

    let mut marker = record.clone();
    if let Some(payload) = marker.payload.as_mut() {
        payload.request = None;
        payload.response = None;
        payload.http_request_headers = None;
        payload.payload_complete = false;
        payload.payload_drop_reason = Some(reason.clone());
    }

    match serde_json::to_string(&marker) {
        Ok(payload) => Some((payload, false, Some(reason))),
        Err(error) => {
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                error = %error,
                "request trace otel: marker serialize failed"
            );
            None
        }
    }
}

fn event_time_from_ms(ms: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(ms)
}

fn schema_name(schema: RequestTraceSchema) -> &'static str {
    match schema {
        RequestTraceSchema::V1 => "dynamo.request.trace.v1",
    }
}

fn event_type_name(event_type: RequestTraceEventType) -> &'static str {
    match event_type {
        RequestTraceEventType::RequestEnd => "request_end",
        RequestTraceEventType::ToolStart => "tool_start",
        RequestTraceEventType::ToolEnd => "tool_end",
        RequestTraceEventType::ToolError => "tool_error",
        RequestTraceEventType::RequestPayload => "request_payload",
    }
}

fn request_id(record: &RequestTraceRecord) -> Option<String> {
    record
        .request
        .as_ref()
        .map(|request| request.request_id.clone())
        .or_else(|| {
            record
                .payload
                .as_ref()
                .map(|payload| payload.request_id.clone())
        })
}

fn model(record: &RequestTraceRecord) -> Option<String> {
    record
        .request
        .as_ref()
        .and_then(|request| request.model.clone())
        .or_else(|| record.payload.as_ref().map(|payload| payload.model.clone()))
}

#[async_trait]
impl RequestTraceSink for OtelRequestTraceSink {
    fn name(&self) -> &'static str {
        "otel"
    }

    async fn emit(&self, record: &RequestTraceRecord) {
        let observed_timestamp = SystemTime::now();
        let start = std::time::Instant::now();
        let payload_result = Self::payload_for_limit(record, self.max_payload_bytes);
        tracing::debug!(
            target: "dynamo.request_trace.otel.serde",
            event_type = event_type_name(record.event_type),
            elapsed_us = start.elapsed().as_micros() as u64,
            payload_len = payload_result.as_ref().map(|(payload, _, _)| payload.len()).unwrap_or(0),
            "OTEL request trace payload serialized"
        );
        let Some((payload, payload_complete, payload_drop_reason)) = payload_result else {
            return;
        };

        let mut otel_record = self.logger.create_log_record();
        otel_record.set_timestamp(event_time_from_ms(record.event_time_unix_ms));
        otel_record.set_observed_timestamp(observed_timestamp);
        otel_record.set_severity_number(Severity::Info);
        otel_record.set_severity_text("INFO");
        otel_record.set_body(AnyValue::String(event_type_name(record.event_type).into()));
        otel_record.add_attribute(
            "schema",
            AnyValue::String(schema_name(record.schema).into()),
        );
        otel_record.add_attribute(
            "event_type",
            AnyValue::String(event_type_name(record.event_type).into()),
        );
        if let Some(request_id) = request_id(record) {
            otel_record.add_attribute("rid", AnyValue::String(request_id.into()));
        }
        if let Some(model) = model(record) {
            otel_record.add_attribute("model", AnyValue::String(model.into()));
        }
        if let Some(payload_record) = record.payload.as_ref() {
            otel_record.add_attribute(
                "endpoint",
                AnyValue::String(payload_record.endpoint.clone().into()),
            );
        }
        otel_record.add_attribute("payload_complete", AnyValue::Boolean(payload_complete));
        if let Some(reason) = payload_drop_reason {
            otel_record.add_attribute("payload_drop_reason", AnyValue::String(reason.into()));
        }
        otel_record.add_attribute("payload", AnyValue::String(payload.into()));

        let _guard = Context::new().attach();
        self.logger.emit(otel_record);
    }

    async fn shutdown(&self) {
        if let Err(error) = self.provider.force_flush() {
            tracing::warn!(
                target: "dynamo_llm::request_trace",
                error = %error,
                "request trace otel: force_flush failed during shutdown"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::request_trace::RequestTraceEventSource;
    use crate::request_trace::RequestTracePayload;

    fn sample_payload_record() -> RequestTraceRecord {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "store": true
        }))
        .unwrap();

        RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::RequestPayload,
            event_time_unix_ms: 1_000,
            event_source: Some(RequestTraceEventSource::Dynamo),
            agent_context: None,
            request: None,
            tool: None,
            payload: Some(RequestTracePayload {
                request_id: "req-otel".to_string(),
                endpoint: "openai.chat_completion".to_string(),
                model: "test-model".to_string(),
                request: Some(Arc::new(request)),
                response: None,
                http_request_headers: None,
                payload_complete: true,
                payload_drop_reason: None,
            }),
        }
    }

    #[test]
    fn payload_over_limit_emits_request_trace_marker() {
        let record = sample_payload_record();
        let (payload, complete, drop_reason) =
            OtelRequestTraceSink::payload_for_limit(&record, 1).unwrap();

        assert!(!complete);
        assert!(drop_reason.unwrap().starts_with("otel_payload_too_large:"));
        let decoded: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(decoded["schema"], "dynamo.request.trace.v1");
        assert_eq!(decoded["event_type"], "request_payload");
        assert_eq!(decoded["payload"]["request_id"], "req-otel");
        assert_eq!(decoded["payload"]["payload_complete"], false);
        assert!(decoded["payload"].get("request").is_none());
        assert!(decoded["payload"].get("response").is_none());
        assert!(
            decoded["payload"]["payload_drop_reason"]
                .as_str()
                .unwrap()
                .starts_with("otel_payload_too_large:")
        );
    }
}
