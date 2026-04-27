// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod bus;
pub mod config;
mod sink;
pub mod stream;
pub mod types;

use crate::context::AgentContext;
use tokio_util::sync::CancellationToken;

pub use config::{AgentTracePolicy, is_enabled, policy};
pub use types::{
    AgentRequestMetrics, AgentTraceRecord, TraceEventSource, TraceEventType, TraceSchema,
    WorkerInfo,
};

pub async fn init_from_env() -> anyhow::Result<()> {
    init_from_env_with_shutdown(CancellationToken::new()).await
}

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
    }

    bus::init(policy.capacity);
    if let Some(path) = policy.jsonl_path.clone() {
        sink::spawn_jsonl_worker_with_shutdown(path, shutdown).await?;
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
        .and_then(|ms| {
            if !ms.is_finite() {
                tracing::debug!(total_time_ms = ms, "invalid agent trace total_time_ms");
                return None;
            }
            Some(
                request
                    .request_received_ms
                    .saturating_add(ms.max(0.0).round() as u64),
            )
        })
        .unwrap_or(request.request_received_ms);

    let record = AgentTraceRecord {
        schema: TraceSchema::V1,
        event_type: TraceEventType::RequestEnd,
        event_time_unix_ms,
        event_source: TraceEventSource::Dynamo,
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

    use super::{
        AgentRequestMetrics, AgentTraceRecord, TraceEventSource, TraceEventType, TraceSchema, bus,
        emit_request_end, sink,
    };
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn test_agent_trace_jsonl_sink_writes_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("agent_trace.jsonl");
        bus::init(16);
        sink::spawn_jsonl_worker_with_shutdown(
            path.display().to_string(),
            CancellationToken::new(),
        )
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

        let mut content = String::new();
        for _ in 0..100 {
            content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            if content.contains("\"event_type\":\"request_end\"")
                && content.contains("\"request_id\":\"req-123\"")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(content.contains("\"event_type\":\"request_end\""));
        assert!(content.contains("\"request_id\":\"req-123\""));
        assert!(content.contains("\"workflow_id\":\"run-1\""));

        let record: AgentTraceRecord = serde_json::from_str(content.lines().next().unwrap())
            .expect("jsonl record should deserialize");
        assert_eq!(record.schema, TraceSchema::V1);
        assert_eq!(record.event_type, TraceEventType::RequestEnd);
        assert_eq!(record.event_source, TraceEventSource::Dynamo);
    }
}
