// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod bus;
pub mod config;
mod sink;
pub mod stream;
pub mod types;

use crate::context::AgentContext;

pub use config::{AgentTracePolicy, is_enabled, policy};
pub use types::{AgentRequestMetrics, AgentTraceRecord, WorkerInfo};

pub async fn init_from_env() -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
    }

    bus::init(policy.capacity);
    if let Some(path) = policy.jsonl_path.clone() {
        sink::spawn_jsonl_worker(path).await?;
    }
    tracing::info!(cap = policy.capacity, "Agent trace initialized");
    Ok(())
}

pub fn publish(rec: AgentTraceRecord) {
    bus::publish(rec);
}

pub fn emit_request_end(agent_context: AgentContext, request: AgentRequestMetrics) {
    let event_time_unix_ms = request
        .total_time_ms
        .map(|ms| {
            request
                .request_received_ms
                .saturating_add(ms.round() as u64)
        })
        .unwrap_or(request.request_received_ms);

    let record = AgentTraceRecord {
        schema: "dynamo.agent.trace.v1".to_string(),
        event_type: "request_end".to_string(),
        event_time_unix_ms,
        event_source: "dynamo".to_string(),
        agent_context,
        request,
    };
    publish(record);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tempfile::tempdir;

    use crate::context::AgentContext;

    use super::{AgentRequestMetrics, bus, emit_request_end, sink};

    #[tokio::test]
    async fn test_agent_trace_jsonl_sink_writes_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("agent_trace.jsonl");
        bus::init(16);
        sink::spawn_jsonl_worker(path.display().to_string())
            .await
            .expect("sink should start");

        emit_request_end(
            AgentContext {
                workflow_type_id: "ms_agent".to_string(),
                workflow_id: "run-1".to_string(),
                program_id: "run-1:agent".to_string(),
                parent_program_id: None,
            },
            AgentRequestMetrics {
                request_id: "req-123".to_string(),
                model: "test-model".to_string(),
                input_tokens: Some(42),
                output_tokens: Some(7),
                cached_tokens: Some(5),
                request_received_ms: 1000,
                ttft_ms: None,
                total_time_ms: Some(25.0),
                queue_depth: None,
                worker: None,
            },
        );

        tokio::time::sleep(Duration::from_millis(50)).await;
        let content = tokio::fs::read_to_string(&path)
            .await
            .expect("sink should write output");
        assert!(content.contains("\"event_type\":\"request_end\""));
        assert!(content.contains("\"request_id\":\"req-123\""));
        assert!(content.contains("\"workflow_id\":\"run-1\""));
    }
}
