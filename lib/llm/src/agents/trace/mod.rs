// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod bus;
pub mod config;
mod integration;
mod publisher;
mod relay;
mod sink;
pub mod stream;
pub mod types;

use crate::agents::context::AgentContext;
use tokio_util::sync::CancellationToken;

pub use config::{AgentTracePolicy, is_enabled, policy};
pub(crate) use integration::{request_metrics, start_tool_event_ingest_from_policy};
pub use publisher::AgentToolEventPublisher;
pub use relay::AgentToolEventRelay;
pub use types::{
    AgentRequestMetrics, AgentToolEvent, AgentToolStatus, AgentTraceRecord, TraceEventSource,
    TraceEventType, TraceSchema, WorkerInfo,
};

pub const DEFAULT_TOOL_EVENTS_TOPIC: &str = "agent-tool-events";
pub(crate) const X_REQUEST_ID_CONTEXT_KEY: &str = "agent_trace.x_request_id";

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
        let options = sink::JsonlSinkOptions::from_policy(&policy);
        sink::spawn_jsonl_worker_with_shutdown(path, options, shutdown).await?;
    }
    tracing::info!(cap = policy.capacity, "Agent trace initialized");
    Ok(())
}

pub fn publish(rec: AgentTraceRecord) {
    bus::publish(rec);
}

pub fn subscribe() -> tokio::sync::broadcast::Receiver<AgentTraceRecord> {
    bus::subscribe()
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
        request: Some(request),
        tool: None,
    };
    publish(record);
}

pub fn emit_tool_event(
    event_type: TraceEventType,
    agent_context: AgentContext,
    tool: AgentToolEvent,
) {
    let record = AgentTraceRecord {
        schema: TraceSchema::V1,
        event_type,
        event_time_unix_ms: current_time_unix_ms(),
        event_source: TraceEventSource::Harness,
        agent_context,
        request: None,
        tool: Some(tool),
    };
    publish_tool_record(record);
}

pub fn publish_tool_record(record: AgentTraceRecord) {
    if let Err(error) = validate_tool_record(&record) {
        tracing::warn!(
            %error,
            event_type = ?record.event_type,
            "dropping invalid agent tool record"
        );
        return;
    }
    publish(record);
}

fn validate_tool_record(record: &AgentTraceRecord) -> anyhow::Result<()> {
    if record.schema != TraceSchema::V1 {
        anyhow::bail!("unsupported agent trace schema: {:?}", record.schema);
    }
    if record.event_source != TraceEventSource::Harness {
        anyhow::bail!(
            "agent tool records must be harness-originated, got {:?}",
            record.event_source
        );
    }
    if !record.event_type.is_tool_event() {
        anyhow::bail!("expected tool event, got {:?}", record.event_type);
    }
    if record.tool.is_none() {
        anyhow::bail!("missing tool payload");
    }
    if record.request.is_some() {
        anyhow::bail!("tool event must not include request metrics");
    }
    Ok(())
}

fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tempfile::tempdir;

    use crate::agents::context::AgentContext;

    use super::{
        AgentRequestMetrics, AgentToolEvent, AgentToolStatus, AgentTraceRecord, TraceEventSource,
        TraceEventType, TraceSchema, bus, emit_request_end, emit_tool_event, sink,
    };
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn test_agent_trace_jsonl_sink_writes_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("agent_trace.jsonl");
        bus::init(16);
        let shutdown = CancellationToken::new();
        sink::spawn_jsonl_worker_with_shutdown(
            path.display().to_string(),
            sink::JsonlSinkOptions {
                buffer_bytes: 128,
                flush_interval: Duration::from_millis(10),
            },
            shutdown.clone(),
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
                x_request_id: Some("llm-call-1".to_string()),
                model: "test-model".to_string(),
                input_tokens: Some(42),
                output_tokens: Some(7),
                cached_tokens: Some(5),
                request_received_ms: 1000,
                prefill_wait_time_ms: None,
                prefill_time_ms: None,
                ttft_ms: None,
                total_time_ms: Some(25.0),
                kv_hit_rate: None,
                kv_transfer_estimated_latency_ms: None,
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
        assert_eq!(
            record.request.unwrap().x_request_id.as_deref(),
            Some("llm-call-1")
        );

        emit_tool_event(
            TraceEventType::ToolEnd,
            AgentContext {
                workflow_type_id: "ms_agent".to_string(),
                workflow_id: "run-1".to_string(),
                program_id: "run-1:agent".to_string(),
                parent_program_id: None,
            },
            AgentToolEvent {
                tool_call_id: "tool-123".to_string(),
                tool_class: "web_search".to_string(),
                status: Some(AgentToolStatus::Succeeded),
                duration_ms: Some(12.5),
                output_tokens: Some(9),
                output_bytes: Some(64),
                tool_name_hash: None,
                error_type: None,
            },
        );

        let mut content = String::new();
        for _ in 0..100 {
            content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            if content.contains("\"event_type\":\"tool_end\"")
                && content.contains("\"tool_call_id\":\"tool-123\"")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(content.contains("\"event_type\":\"tool_end\""));
        assert!(content.contains("\"tool_call_id\":\"tool-123\""));
        assert!(content.contains("\"tool_class\":\"web_search\""));

        let record: AgentTraceRecord = serde_json::from_str(content.lines().nth(1).unwrap())
            .expect("jsonl record should deserialize");
        assert_eq!(record.schema, TraceSchema::V1);
        assert_eq!(record.event_type, TraceEventType::ToolEnd);
        assert_eq!(record.event_source, TraceEventSource::Harness);
        assert!(record.request.is_none());
        assert_eq!(
            record.tool.unwrap().status,
            Some(AgentToolStatus::Succeeded)
        );

        shutdown.cancel();
    }
}
