// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{SystemTime, UNIX_EPOCH};

use crate::protocols::common::extensions::AgentContext;
use crate::protocols::common::timing::RequestTracker;
use crate::request_trace::{RequestReplayMetrics, RequestTraceEventSource};

use super::{
    RequestTraceEventType, RequestTraceMetrics, RequestTraceRecord, RequestTraceSchema, publish,
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

fn sanitize_request(request: &mut RequestTraceMetrics) {
    request.prefill_wait_time_ms = sanitize_finite(request.prefill_wait_time_ms);
    request.prefill_time_ms = sanitize_finite(request.prefill_time_ms);
    request.ttft_ms = sanitize_finite(request.ttft_ms);
    request.total_time_ms = sanitize_finite(request.total_time_ms);
    request.avg_itl_ms = sanitize_finite(request.avg_itl_ms);
    request.kv_hit_rate = sanitize_finite(request.kv_hit_rate);
    request.kv_transfer_estimated_latency_ms =
        sanitize_finite(request.kv_transfer_estimated_latency_ms);
}

fn event_time_unix_ms_from_request(request: &RequestTraceMetrics) -> u64 {
    request
        .request_received_ms
        .map_or_else(unix_time_ms, |received_ms| {
            request
                .total_time_ms
                .map(|ms| received_ms.saturating_add(ms.max(0.0).round() as u64))
                .unwrap_or(received_ms)
        })
}

pub(crate) fn emit_request_end(
    request_id: String,
    tracker: &RequestTracker,
    replay: RequestReplayMetrics,
) {
    let request_received_ms = tracker.request_received_epoch_ms();
    let event_time_unix_ms = tracker
        .total_time_ms()
        .map_or_else(unix_time_ms, |elapsed| {
            request_received_ms.saturating_add(elapsed.max(0.0).round() as u64)
        });

    let request = RequestTraceMetrics {
        request_id,
        x_request_id: None,
        model: None,
        input_tokens: None,
        output_tokens: Some(tracker.osl_tokens()),
        cached_tokens: None,
        request_received_ms: Some(request_received_ms),
        prefill_wait_time_ms: None,
        prefill_time_ms: None,
        ttft_ms: None,
        total_time_ms: None,
        avg_itl_ms: None,
        kv_hit_rate: None,
        kv_transfer_estimated_latency_ms: None,
        queue_depth: None,
        worker: None,
        replay: Some(replay),
        finish_reason_metadata: None,
    };

    publish(RequestTraceRecord {
        schema: RequestTraceSchema::V1,
        event_type: RequestTraceEventType::RequestEnd,
        event_time_unix_ms,
        event_source: None,
        agent_context: None,
        request: Some(request),
        tool: None,
    });
}

pub(crate) fn emit_agent_request_end(
    agent_context: AgentContext,
    mut request: RequestTraceMetrics,
) {
    sanitize_request(&mut request);
    publish(RequestTraceRecord {
        schema: RequestTraceSchema::V1,
        event_type: RequestTraceEventType::RequestEnd,
        event_time_unix_ms: event_time_unix_ms_from_request(&request),
        event_source: Some(RequestTraceEventSource::Dynamo),
        agent_context: Some(agent_context),
        request: Some(request),
        tool: None,
    });
}

pub(crate) fn publish_tool_record(record: RequestTraceRecord) {
    if let Err(error) = validate_tool_record(&record) {
        tracing::warn!(
            %error,
            event_type = ?record.event_type,
            "dropping invalid request trace tool record"
        );
        return;
    }

    publish(record);
}

pub(crate) fn validate_tool_record(record: &RequestTraceRecord) -> anyhow::Result<()> {
    if record.schema != RequestTraceSchema::V1 {
        anyhow::bail!("unsupported request trace schema: {:?}", record.schema);
    }
    if record.event_source != Some(RequestTraceEventSource::Harness) {
        anyhow::bail!(
            "request trace tool records must be harness-originated, got {:?}",
            record.event_source
        );
    }
    if !record.event_type.is_tool_event() {
        anyhow::bail!("expected tool event, got {:?}", record.event_type);
    }
    if record.agent_context.is_none() {
        anyhow::bail!("missing agent_context");
    }
    let Some(tool) = record.tool.as_ref() else {
        anyhow::bail!("missing tool payload");
    };
    if tool.duration_ms.is_some_and(|value| !value.is_finite()) {
        anyhow::bail!("tool duration_ms must be finite");
    }
    if record.request.is_some() {
        anyhow::bail!("tool event must not include request metrics");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_trace::{BUS, RequestTraceToolEvent};

    #[tokio::test]
    async fn emits_tracker_timing_lengths_and_hashes() {
        BUS.init(16);
        let mut rx = BUS.subscribe();
        let tracker = RequestTracker::new();
        tracker.record_osl(7);
        tracker.record_finish();

        emit_request_end(
            "req-1".to_string(),
            &tracker,
            RequestReplayMetrics {
                trace_block_size: 2,
                input_length: 3,
                input_sequence_hashes: vec![11, 22],
            },
        );

        let record = loop {
            let record = rx.recv().await.unwrap();
            if record
                .request
                .as_ref()
                .is_some_and(|request| request.request_id == "req-1")
            {
                break record;
            }
        };
        let request = record.request.as_ref().expect("request payload");
        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.output_tokens, Some(7));
        assert_eq!(
            request.request_received_ms,
            Some(tracker.request_received_epoch_ms())
        );
        assert!(
            record.event_time_unix_ms
                >= request
                    .request_received_ms
                    .expect("request received timestamp")
        );
        assert_eq!(
            request
                .replay
                .as_ref()
                .expect("replay metrics")
                .input_length,
            3
        );
    }

    #[test]
    fn rejects_non_finite_tool_duration() {
        let mut record = RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::ToolEnd,
            event_time_unix_ms: 2_000,
            event_source: Some(RequestTraceEventSource::Harness),
            agent_context: Some(AgentContext {
                trajectory_id: "root".to_string(),
                parent_trajectory_id: None,
                trajectory_final: None,
                kv_hints: None,
            }),
            request: None,
            tool: Some(RequestTraceToolEvent {
                tool_call_id: "tool-1".to_string(),
                tool_class: "web_search".to_string(),
                started_at_unix_ms: None,
                ended_at_unix_ms: None,
                duration_ms: Some(f64::NAN),
                status: None,
                output_bytes: None,
                output_tokens: None,
                tool_name_hash: None,
                error_type: None,
            }),
        };

        let err = validate_tool_record(&record).expect_err("NaN duration should fail validation");
        assert!(err.to_string().contains("tool duration_ms must be finite"));

        record.tool.as_mut().expect("tool payload").duration_ms = Some(f64::INFINITY);
        let err =
            validate_tool_record(&record).expect_err("infinite duration should fail validation");
        assert!(err.to_string().contains("tool duration_ms must be finite"));
    }
}
