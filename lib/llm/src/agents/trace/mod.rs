// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
mod integration;
pub mod types;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio_util::sync::CancellationToken;

use crate::agents::context::AgentContext;
use crate::telemetry::bus::TelemetryBus;
use crate::telemetry::jsonl::{JsonlSinkOptions, spawn_jsonl_worker_with_shutdown};

pub use config::{AgentTracePolicy, is_enabled, policy};
pub(crate) use integration::request_metrics;
pub use types::{
    AgentRequestMetrics, AgentTraceRecord, TraceEventSource, TraceEventType, TraceSchema,
    WorkerInfo,
};

pub(crate) const X_REQUEST_ID_CONTEXT_KEY: &str = "agent_trace.x_request_id";

static BUS: TelemetryBus<AgentTraceRecord> = TelemetryBus::new();
static JSONL_WORKER_STARTED: AtomicBool = AtomicBool::new(false);

pub async fn init_from_env() -> anyhow::Result<()> {
    init_from_env_with_shutdown(CancellationToken::new()).await
}

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
    }

    BUS.init(policy.capacity);
    if let Some(path) = policy.jsonl_path.clone()
        && JSONL_WORKER_STARTED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    {
        if let Err(err) = spawn_jsonl_worker_with_shutdown(
            BUS.subscribe(),
            path,
            JsonlSinkOptions {
                buffer_bytes: policy.jsonl_buffer_bytes,
                flush_interval: Duration::from_millis(policy.jsonl_flush_interval_ms.max(1)),
            },
            shutdown,
        )
        .await
        {
            JSONL_WORKER_STARTED.store(false, Ordering::Release);
            return Err(err);
        }
    }

    tracing::info!(capacity = policy.capacity, "Agent trace initialized");
    Ok(())
}

pub fn publish(rec: AgentTraceRecord) {
    BUS.publish(rec);
}

pub fn subscribe() -> tokio::sync::broadcast::Receiver<AgentTraceRecord> {
    BUS.subscribe()
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn sanitize_finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

pub fn emit_request_end(agent_context: AgentContext, request: AgentRequestMetrics) {
    let mut request = request;
    request.prefill_wait_time_ms = sanitize_finite(request.prefill_wait_time_ms);
    request.prefill_time_ms = sanitize_finite(request.prefill_time_ms);
    request.ttft_ms = sanitize_finite(request.ttft_ms);
    request.total_time_ms = sanitize_finite(request.total_time_ms);
    request.kv_hit_rate = sanitize_finite(request.kv_hit_rate);
    request.kv_transfer_estimated_latency_ms =
        sanitize_finite(request.kv_transfer_estimated_latency_ms);

    let event_time_unix_ms = request
        .request_received_ms
        .map_or_else(unix_time_ms, |received_ms| {
            request
                .total_time_ms
                .map(|ms| received_ms.saturating_add(ms.max(0.0).round() as u64))
                .unwrap_or(received_ms)
        });

    publish(AgentTraceRecord {
        schema: TraceSchema::V1,
        event_type: TraceEventType::RequestEnd,
        event_time_unix_ms,
        event_source: TraceEventSource::Dynamo,
        agent_context,
        request: Some(request),
    });
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;

    use crate::agents::context::AgentContext;
    use crate::telemetry::jsonl::{JsonlSinkOptions, spawn_jsonl_worker_with_shutdown};

    use super::{
        AgentRequestMetrics, AgentTraceRecord, BUS, TraceEventSource, TraceEventType, TraceSchema,
        emit_request_end,
    };

    #[tokio::test]
    async fn test_agent_trace_jsonl_sink_writes_request_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("agent_trace.jsonl");
        BUS.init(16);
        let shutdown = CancellationToken::new();
        spawn_jsonl_worker_with_shutdown(
            BUS.subscribe(),
            path.display().to_string(),
            JsonlSinkOptions {
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
                request_received_ms: Some(1000),
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

        let wrapper = content
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|wrapper| {
                wrapper
                    .get("event")
                    .and_then(|event| event.get("request"))
                    .and_then(|request| request.get("request_id"))
                    .and_then(serde_json::Value::as_str)
                    == Some("req-123")
            })
            .expect("expected req-123 record in jsonl");
        let record: AgentTraceRecord = serde_json::from_value(wrapper["event"].clone())
            .expect("jsonl event should deserialize");
        assert_eq!(record.schema, TraceSchema::V1);
        assert_eq!(record.event_type, TraceEventType::RequestEnd);
        assert_eq!(record.event_source, TraceEventSource::Dynamo);
        assert_eq!(
            record.request.unwrap().x_request_id.as_deref(),
            Some("llm-call-1")
        );

        shutdown.cancel();
    }

    #[tokio::test]
    async fn test_emit_request_end_sanitizes_non_finite_metrics() {
        BUS.init(16);
        let mut rx = BUS.subscribe();

        emit_request_end(
            AgentContext {
                workflow_type_id: "ms_agent".to_string(),
                workflow_id: "run-non-finite".to_string(),
                program_id: "run-non-finite:agent".to_string(),
                parent_program_id: None,
            },
            AgentRequestMetrics {
                request_id: "req-non-finite".to_string(),
                x_request_id: None,
                model: "test-model".to_string(),
                input_tokens: None,
                output_tokens: None,
                cached_tokens: None,
                request_received_ms: Some(1000),
                prefill_wait_time_ms: Some(f64::NAN),
                prefill_time_ms: Some(f64::INFINITY),
                ttft_ms: Some(f64::NEG_INFINITY),
                total_time_ms: Some(f64::NAN),
                kv_hit_rate: Some(f64::INFINITY),
                kv_transfer_estimated_latency_ms: Some(f64::NEG_INFINITY),
                queue_depth: None,
                worker: None,
            },
        );

        let record = rx.recv().await.expect("trace record should publish");
        assert_eq!(record.event_time_unix_ms, 1000);
        let request = record.request.expect("request metrics should be present");
        assert_eq!(request.prefill_wait_time_ms, None);
        assert_eq!(request.prefill_time_ms, None);
        assert_eq!(request.ttft_ms, None);
        assert_eq!(request.total_time_ms, None);
        assert_eq!(request.kv_hit_rate, None);
        assert_eq!(request.kv_transfer_estimated_latency_ms, None);
    }
}
