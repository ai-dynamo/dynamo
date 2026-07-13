// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::protocols::common::extensions::AgentContext;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTraceRecord {
    pub schema: RequestTraceSchema,
    pub event_type: RequestTraceEventType,
    pub event_time_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_source: Option<RequestTraceEventSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_context: Option<AgentContext>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestTraceMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<RequestTraceToolEvent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<RequestTracePayload>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RequestTraceSchema {
    #[serde(rename = "dynamo.request.trace.v1")]
    V1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RequestTraceEventType {
    #[serde(rename = "request_end")]
    RequestEnd,
    #[serde(rename = "tool_start")]
    ToolStart,
    #[serde(rename = "tool_end")]
    ToolEnd,
    #[serde(rename = "tool_error")]
    ToolError,
    #[serde(rename = "request_payload")]
    RequestPayload,
}

impl RequestTraceEventType {
    pub fn is_tool_event(self) -> bool {
        matches!(self, Self::ToolStart | Self::ToolEnd | Self::ToolError)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTracePayload {
    pub request_id: String,
    pub endpoint: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>,
    /// Allowlisted HTTP request headers (`DYN_REQUEST_TRACE_HTTP_HEADER_CAPTURE_LIST`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_request_headers: Option<Arc<BTreeMap<String, String>>>,
    pub payload_complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_drop_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RequestTraceEventSource {
    #[serde(rename = "dynamo")]
    Dynamo,
    #[serde(rename = "harness")]
    Harness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTraceMetrics {
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_received_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_wait_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_itl_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_hit_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_transfer_estimated_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_depth: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<RequestTraceWorkerInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay: Option<RequestReplayMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason_metadata: Option<FinishReasonMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestReplayMetrics {
    pub trace_block_size: usize,
    pub input_length: usize,
    pub input_sequence_hashes: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestTraceWorkerInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_dp_rank: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FinishReasonMetadata {
    /// Final OpenAI-compatible finish reason after parser/post-processing rewrites.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<dynamo_protocols::types::FinishReason>,

    /// Raw backend finish reason before parser/post-processing rewrites.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_finish_reason: Option<String>,

    /// Backend-provided stop condition that caused `finish_reason=stop`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<dynamo_protocols::types::StopReason>,

    /// Complete tool calls emitted by the model. Arguments are intentionally
    /// omitted so request traces remain metadata, not payload logs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallMetadata>,

    /// Per-choice finish metadata for multi-choice requests. The top-level
    /// fields remain as a compact summary for the common single-choice case.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<ChoiceFinishReasonMetadata>,
}

impl FinishReasonMetadata {
    pub fn is_empty(&self) -> bool {
        self.finish_reason.is_none()
            && self.backend_finish_reason.is_none()
            && self.stop_reason.is_none()
            && self.tool_calls.is_empty()
            && self.choices.is_empty()
    }

    pub fn record_choice_finish_reason(
        &mut self,
        choice_index: u32,
        finish_reason: dynamo_protocols::types::FinishReason,
    ) {
        self.choice_metadata_mut(choice_index).finish_reason = Some(finish_reason);
    }

    pub fn record_choice_backend_finish_reason(
        &mut self,
        choice_index: u32,
        backend_finish_reason: Option<String>,
        stop_reason: Option<dynamo_protocols::types::StopReason>,
    ) {
        if backend_finish_reason.is_none() && stop_reason.is_none() {
            return;
        }
        let choice = self.choice_metadata_mut(choice_index);
        if let Some(backend_finish_reason) = backend_finish_reason {
            choice.backend_finish_reason = Some(backend_finish_reason);
        }
        if let Some(stop_reason) = stop_reason {
            choice.stop_reason = Some(stop_reason);
        }
    }

    fn choice_metadata_mut(&mut self, choice_index: u32) -> &mut ChoiceFinishReasonMetadata {
        if let Some(position) = self
            .choices
            .iter()
            .position(|choice| choice.choice_index == choice_index)
        {
            return &mut self.choices[position];
        }

        let position = self.choices.len();
        self.choices.push(ChoiceFinishReasonMetadata {
            choice_index,
            finish_reason: None,
            backend_finish_reason: None,
            stop_reason: None,
        });
        &mut self.choices[position]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallMetadata {
    pub choice_index: u32,
    pub tool_call_index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChoiceFinishReasonMetadata {
    pub choice_index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<dynamo_protocols::types::FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<dynamo_protocols::types::StopReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTraceToolEvent {
    pub tool_call_id: String,
    pub tool_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<RequestTraceToolStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTraceToolEventIngress {
    pub schema: RequestTraceSchema,
    pub event_type: RequestTraceEventType,
    pub event_time_unix_ms: u64,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,
    pub tool: RequestTraceToolEvent,
}

impl From<RequestTraceToolEventIngress> for RequestTraceRecord {
    fn from(ingress: RequestTraceToolEventIngress) -> Self {
        Self {
            schema: ingress.schema,
            event_type: ingress.event_type,
            event_time_unix_ms: ingress.event_time_unix_ms,
            event_source: Some(RequestTraceEventSource::Harness),
            agent_context: Some(AgentContext {
                session_id: ingress.session_id,
                parent_session_id: ingress.parent_session_id,
                session_final: None,
                kv_hints: None,
            }),
            request: None,
            tool: Some(ingress.tool),
            payload: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RequestTraceToolStatus {
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "succeeded", alias = "ok", alias = "success")]
    Succeeded,
    #[serde(rename = "error", alias = "failed")]
    Error,
    #[serde(rename = "cancelled", alias = "canceled", alias = "timeout")]
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_trace_has_only_replay_fields() {
        let record = RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::RequestEnd,
            event_time_unix_ms: 1_100,
            event_source: None,
            agent_context: None,
            request: Some(RequestTraceMetrics {
                request_id: "req-1".to_string(),
                x_request_id: None,
                model: None,
                input_tokens: None,
                output_tokens: Some(4),
                cached_tokens: None,
                request_received_ms: Some(1_000),
                prefill_wait_time_ms: None,
                prefill_time_ms: None,
                ttft_ms: None,
                total_time_ms: None,
                avg_itl_ms: None,
                kv_hit_rate: None,
                kv_transfer_estimated_latency_ms: None,
                queue_depth: None,
                worker: None,
                replay: Some(RequestReplayMetrics {
                    trace_block_size: 2,
                    input_length: 4,
                    input_sequence_hashes: vec![11, 22],
                }),
                finish_reason_metadata: None,
            }),
            tool: None,
            payload: None,
        };

        let value = serde_json::to_value(record).unwrap();
        assert_eq!(value["schema"], "dynamo.request.trace.v1");
        assert_eq!(value["event_type"], "request_end");
        assert!(value.get("event_source").is_none());
        assert!(value.get("agent_context").is_none());
        assert!(value.get("tool").is_none());
        assert!(value.get("payload").is_none());
        assert!(value["request"].get("model").is_none());
        assert!(value["request"].get("finish_reason_metadata").is_none());
    }

    #[test]
    fn request_payload_record_serializes_unified_schema() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "store": true
        }))
        .unwrap();
        let response: NvCreateChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }]
        }))
        .unwrap();
        let record = RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::RequestPayload,
            event_time_unix_ms: 1_100,
            event_source: Some(RequestTraceEventSource::Dynamo),
            agent_context: None,
            request: None,
            tool: None,
            payload: Some(RequestTracePayload {
                request_id: "req-1".to_string(),
                endpoint: "openai.chat_completion".to_string(),
                model: "test-model".to_string(),
                request: Some(Arc::new(request)),
                response: Some(Arc::new(response)),
                http_request_headers: None,
                payload_complete: true,
                payload_drop_reason: None,
            }),
        };

        let value = serde_json::to_value(record).unwrap();
        assert_eq!(value["schema"], "dynamo.request.trace.v1");
        assert_eq!(value["event_type"], "request_payload");
        assert_eq!(value["payload"]["request_id"], "req-1");
        assert_eq!(value["payload"]["endpoint"], "openai.chat_completion");
        assert!(value["payload"].get("requested_streaming").is_none());
        assert_eq!(value["payload"]["payload_complete"], true);
        assert_eq!(value["payload"]["request"]["model"], "test-model");
        assert_eq!(
            value["payload"]["response"]["choices"][0]["message"]["content"],
            "hi"
        );
        assert!(value.get("schema_version").is_none());
        assert!(value.get("audit_complete").is_none());
    }

    #[test]
    fn payload_serializes_http_request_headers_and_omits_when_absent() {
        let mut headers = BTreeMap::new();
        headers.insert("x-request-id".to_string(), "abc-123".to_string());
        let record = RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::RequestPayload,
            event_time_unix_ms: 1_100,
            event_source: Some(RequestTraceEventSource::Dynamo),
            agent_context: None,
            request: None,
            tool: None,
            payload: Some(RequestTracePayload {
                request_id: "req-h".to_string(),
                endpoint: "openai.chat_completion".to_string(),
                model: "test-model".to_string(),
                request: None,
                response: None,
                http_request_headers: Some(Arc::new(headers)),
                payload_complete: true,
                payload_drop_reason: None,
            }),
        };

        let value = serde_json::to_value(&record).unwrap();
        assert_eq!(
            value["payload"]["http_request_headers"]["x-request-id"],
            "abc-123"
        );

        let roundtrip: RequestTraceRecord = serde_json::from_value(value).unwrap();
        let headers = roundtrip
            .payload
            .as_ref()
            .and_then(|payload| payload.http_request_headers.as_ref())
            .expect("headers survive deserialization");
        assert_eq!(
            headers.get("x-request-id").map(String::as_str),
            Some("abc-123")
        );

        let mut bare = record;
        if let Some(payload) = bare.payload.as_mut() {
            payload.http_request_headers = None;
        }
        let value = serde_json::to_value(&bare).unwrap();
        assert!(value["payload"].get("http_request_headers").is_none());
    }

    #[test]
    fn request_trace_tool_status_accepts_documented_aliases() {
        for (input, expected) in [
            ("running", RequestTraceToolStatus::Running),
            ("succeeded", RequestTraceToolStatus::Succeeded),
            ("ok", RequestTraceToolStatus::Succeeded),
            ("success", RequestTraceToolStatus::Succeeded),
            ("error", RequestTraceToolStatus::Error),
            ("failed", RequestTraceToolStatus::Error),
            ("cancelled", RequestTraceToolStatus::Cancelled),
            ("canceled", RequestTraceToolStatus::Cancelled),
            ("timeout", RequestTraceToolStatus::Cancelled),
        ] {
            let parsed: RequestTraceToolStatus =
                serde_json::from_str(&format!("\"{input}\"")).unwrap();
            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn empty_choice_backend_finish_reason_is_ignored() {
        let mut metadata = FinishReasonMetadata::default();

        metadata.record_choice_backend_finish_reason(0, None, None);

        assert!(metadata.is_empty());
        assert!(metadata.choices.is_empty());
    }
}
