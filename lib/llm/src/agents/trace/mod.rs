// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod config;
mod integration;
pub mod sink;
pub mod types;

use std::time::{SystemTime, UNIX_EPOCH};

use tokio_util::sync::CancellationToken;

use crate::agents::context::AgentContext;
use crate::telemetry::bus::TelemetryBus;

pub use config::{AgentTracePolicy, is_enabled, policy};
pub(crate) use integration::{record_llm_metric_tokens, request_metrics};
pub use types::{
    AgentRequestMetrics, AgentTraceRecord, TraceEventSource, TraceEventType, TraceSchema,
    WorkerInfo,
};

pub(crate) const X_REQUEST_ID_CONTEXT_KEY: &str = "agent_trace.x_request_id";

static BUS: TelemetryBus<AgentTraceRecord> = TelemetryBus::new();

pub async fn init_from_env() -> anyhow::Result<()> {
    init_from_env_with_shutdown(CancellationToken::new()).await
}

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        return Ok(());
    }

    BUS.init(policy.capacity);
    sink::spawn_workers_from_env(shutdown).await?;

    tracing::info!(
        capacity = policy.capacity,
        sinks = ?policy.sinks,
        "Agent trace initialized"
    );
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
    request.avg_itl_ms = sanitize_finite(request.avg_itl_ms);
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
    use crate::agents::context::AgentContext;

    use super::{AgentRequestMetrics, BUS, emit_request_end};

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
                avg_itl_ms: Some(f64::INFINITY),
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
        assert_eq!(request.avg_itl_ms, None);
        assert_eq!(request.kv_hit_rate, None);
        assert_eq!(request.kv_transfer_estimated_latency_ms, None);
    }
}
