// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{SystemTime, UNIX_EPOCH};

use crate::agents::context::AgentContext;

use super::{
    AgentRequestMetrics, AgentTraceRecord, TraceEventSource, TraceEventType, TraceSchema, publish,
};

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn sanitize_finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

pub fn emit_request_end(agent_context: AgentContext, mut request: AgentRequestMetrics) {
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
        tool: None,
    });
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

pub(crate) fn validate_tool_record(record: &AgentTraceRecord) -> anyhow::Result<()> {
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

#[cfg(test)]
mod tests {
    use crate::agents::context::AgentContext;

    use super::{
        AgentRequestMetrics, AgentTraceRecord, TraceEventSource, TraceEventType, TraceSchema,
        emit_request_end, publish_tool_record,
    };
    use crate::agents::trace::{AgentToolEvent, AgentToolStatus, BUS};

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
                replay: None,
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

    #[tokio::test]
    async fn test_publish_tool_record_accepts_valid_tool_record() {
        BUS.init(16);
        let mut rx = BUS.subscribe();

        publish_tool_record(AgentTraceRecord {
            schema: TraceSchema::V1,
            event_type: TraceEventType::ToolEnd,
            event_time_unix_ms: 2000,
            event_source: TraceEventSource::Harness,
            agent_context: AgentContext {
                workflow_type_id: "ms_agent".to_string(),
                workflow_id: "run-1".to_string(),
                program_id: "run-1:agent".to_string(),
                parent_program_id: None,
            },
            request: None,
            tool: Some(AgentToolEvent {
                tool_call_id: "tool-123".to_string(),
                tool_class: "web_search".to_string(),
                started_at_unix_ms: None,
                ended_at_unix_ms: None,
                status: Some(AgentToolStatus::Succeeded),
                duration_ms: Some(12.5),
                output_tokens: Some(9),
                output_bytes: Some(64),
                tool_name_hash: None,
                error_type: None,
            }),
        });

        let record = rx.recv().await.expect("tool record should publish");
        assert_eq!(record.schema, TraceSchema::V1);
        assert_eq!(record.event_type, TraceEventType::ToolEnd);
        assert_eq!(record.event_source, TraceEventSource::Harness);
        assert!(record.request.is_none());
        assert_eq!(
            record.tool.unwrap().status,
            Some(AgentToolStatus::Succeeded)
        );
    }
}
