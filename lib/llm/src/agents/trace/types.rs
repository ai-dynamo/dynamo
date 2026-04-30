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
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TraceEventSource {
    #[serde(rename = "dynamo")]
    Dynamo,
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
