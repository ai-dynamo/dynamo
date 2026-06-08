// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::agents::trace::AgentReplayMetrics;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestTraceRecord {
    pub schema: RequestTraceSchema,
    pub event_type: RequestTraceEventType,
    pub event_time_unix_ms: u64,
    pub request: RequestTraceMetrics,
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestTraceMetrics {
    pub request_id: String,
    pub request_received_ms: u64,
    pub output_tokens: u64,
    pub replay: AgentReplayMetrics,
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
            request: RequestTraceMetrics {
                request_id: "req-1".to_string(),
                request_received_ms: 1_000,
                output_tokens: 4,
                replay: AgentReplayMetrics {
                    trace_block_size: 2,
                    input_length: 4,
                    input_sequence_hashes: vec![11, 22],
                },
            },
        };

        let value = serde_json::to_value(record).unwrap();
        assert_eq!(value["schema"], "dynamo.request.trace.v1");
        assert_eq!(value["event_type"], "request_end");
        assert!(value.get("agent_context").is_none());
        assert!(value.get("tool").is_none());
        assert!(value["request"].get("model").is_none());
        assert!(value["request"].get("finish_reason_metadata").is_none());
        assert!(value["request"].get("payload").is_none());
    }
}
