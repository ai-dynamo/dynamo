// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::agents::context::AgentContext;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTraceRecord {
    pub schema: TraceSchema,
    pub event_type: TraceEventType,
    pub event_time_unix_ms: u64,
    pub event_source: TraceEventSource,
    pub agent_context: AgentContext,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<AgentRequestMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<AgentToolEvent>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TraceSchema {
    #[serde(rename = "dynamo.agent.trace.v1")]
    V1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TraceEventType {
    #[serde(rename = "request_end")]
    RequestEnd,
    #[serde(rename = "tool_start")]
    ToolStart,
    #[serde(rename = "tool_end")]
    ToolEnd,
    #[serde(rename = "tool_error")]
    ToolError,
}

impl TraceEventType {
    pub fn is_tool_event(self) -> bool {
        matches!(self, Self::ToolStart | Self::ToolEnd | Self::ToolError)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TraceEventSource {
    #[serde(rename = "dynamo")]
    Dynamo,
    #[serde(rename = "harness", alias = "ms_agent")]
    Harness,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRequestMetrics {
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_request_id: Option<String>,
    pub model: String,
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
    pub worker: Option<WorkerInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay: Option<AgentReplayMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub io_text: Option<AgentIoText>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentReplayMetrics {
    pub trace_block_size: usize,
    pub input_length: usize,
    pub input_sequence_hashes: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentIoText {
    pub input: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_dp_rank: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolEvent {
    pub tool_call_id: String,
    pub tool_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<AgentToolStatus>,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentToolStatus {
    #[serde(rename = "running")]
    Running,
    #[serde(rename = "succeeded", alias = "ok", alias = "success")]
    Succeeded,
    #[serde(rename = "error", alias = "failed")]
    Error,
    #[serde(rename = "cancelled", alias = "timeout", alias = "canceled")]
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::{AgentIoText, AgentRequestMetrics};

    fn minimal_request(io_text: Option<AgentIoText>) -> AgentRequestMetrics {
        AgentRequestMetrics {
            request_id: "req-1".to_string(),
            x_request_id: None,
            model: "test-model".to_string(),
            input_tokens: None,
            output_tokens: None,
            cached_tokens: None,
            request_received_ms: None,
            prefill_wait_time_ms: None,
            prefill_time_ms: None,
            ttft_ms: None,
            total_time_ms: None,
            avg_itl_ms: None,
            kv_hit_rate: None,
            kv_transfer_estimated_latency_ms: None,
            queue_depth: None,
            worker: None,
            replay: None,
            io_text,
        }
    }

    #[test]
    fn request_metrics_omits_io_text_by_default() {
        let json = serde_json::to_value(minimal_request(None)).unwrap();
        assert!(json.get("io_text").is_none());
    }

    #[test]
    fn request_metrics_serializes_io_text_when_present() {
        let json = serde_json::to_value(minimal_request(Some(AgentIoText {
            input: "rendered input".to_string(),
            output: "decoded output".to_string(),
        })))
        .unwrap();

        assert_eq!(json["io_text"]["input"], "rendered input");
        assert_eq!(json["io_text"]["output"], "decoded output");
    }
}
