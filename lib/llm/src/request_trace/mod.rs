// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod agent_context;
pub mod config;
mod integration;
mod otel_sink;
pub mod payload;
pub(crate) mod payload_stream;
mod record;
mod replay;
pub mod sink;
mod tool_relay;
pub mod types;

use tokio_util::sync::CancellationToken;

use dynamo_runtime::DistributedRuntime;

use crate::local_model::LocalModel;
use crate::telemetry::bus::TelemetryBus;

pub use agent_context::SharedFinishReasonMetadata;
pub(crate) use agent_context::{
    AgentContextTraceState, build_agent_context_trace_state, into_owned_replay_metrics,
    record_backend_finish_reason_metadata, record_chat_finish_reason_metadata,
    record_completion_finish_reason_metadata, record_llm_metric_tokens, request_metrics,
    request_metrics_from_agent_state, start_request_trace_tool_event_ingest,
};
pub use config::{
    RequestTraceFileFormat, RequestTracePolicy, RequestTraceRecordKind, RequestTraceSinkKind,
    is_enabled, policy,
};
pub(crate) use integration::{
    build_request_end_trace_state, finish_reason_metadata_handle, wrap_chat_request_end_stream,
    wrap_completion_request_end_stream,
};
pub(crate) use record::{publish_tool_record, validate_tool_record};
pub(crate) use replay::replay_metrics;
pub use types::{
    ChoiceFinishReasonMetadata, FinishReasonMetadata, RequestReplayMetrics,
    RequestTraceEventSource, RequestTraceEventType, RequestTraceMetrics, RequestTracePayload,
    RequestTraceRecord, RequestTraceSchema, RequestTraceToolEvent, RequestTraceToolEventIngress,
    RequestTraceToolStatus, RequestTraceWorkerInfo, ToolCallMetadata,
};

static BUS: TelemetryBus<RequestTraceRecord> = TelemetryBus::new();
static METADATA_BUS: TelemetryBus<RequestTraceRecord> = TelemetryBus::new();
static PAYLOAD_BUS: TelemetryBus<RequestTraceRecord> = TelemetryBus::new();

/// Receiver for both request-trace delivery lanes.
///
/// Payload records use a separate bounded lane so a burst of large payloads
/// cannot evict request and tool metadata before a sink gets to observe it.
/// The lanes are selected fairly to avoid starvation. Event timestamps remain
/// the source of truth for ordering in the unified output stream.
pub(crate) struct RequestTraceReceiver {
    metadata: tokio::sync::broadcast::Receiver<RequestTraceRecord>,
    payload: tokio::sync::broadcast::Receiver<RequestTraceRecord>,
}

impl RequestTraceReceiver {
    pub(crate) async fn recv(
        &mut self,
    ) -> Result<RequestTraceRecord, tokio::sync::broadcast::error::RecvError> {
        tokio::select! {
            result = self.metadata.recv() => result,
            result = self.payload.recv() => result,
        }
    }

    pub(crate) fn try_recv(
        &mut self,
    ) -> Result<RequestTraceRecord, tokio::sync::broadcast::error::TryRecvError> {
        use tokio::sync::broadcast::error::TryRecvError;

        match self.metadata.try_recv() {
            Err(TryRecvError::Empty | TryRecvError::Closed) => self.payload.try_recv(),
            result => result,
        }
    }
}

pub const DEFAULT_TOOL_EVENTS_TOPIC: &str = "agent-tool-events";
pub(crate) const X_REQUEST_ID_CONTEXT_KEY: &str = "request_trace.x_request_id";

pub async fn init_from_env_with_shutdown(shutdown: CancellationToken) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled {
        config::mark_capture_inactive();
        return Ok(());
    }

    config::mark_capture_inactive();

    if policy.tool_events_zmq_endpoint.is_some()
        && policy.emit_tool_records()
        && policy.sinks.is_empty()
    {
        tracing::warn!(
            tool_events_zmq_endpoint = ?policy.tool_events_zmq_endpoint,
            "request trace tool events are enabled but no local trace sinks are configured; set DYN_REQUEST_TRACE_SINKS to write local trace records"
        );
    }

    BUS.init(policy.capacity);
    METADATA_BUS.init(policy.capacity);
    PAYLOAD_BUS.init(policy.capacity);
    sink::spawn_workers_from_env(shutdown).await?;
    config::mark_capture_active();
    tracing::info!(
        capacity = policy.capacity,
        sinks = ?policy.sink_names(),
        file_format = policy.file_format.as_str(),
        records = ?policy.record_names(),
        "Request trace initialized"
    );
    Ok(())
}

pub(crate) async fn start_tool_event_ingest_from_policy(
    drt: DistributedRuntime,
    local_model: &LocalModel,
) -> anyhow::Result<()> {
    let policy = policy();
    if !policy.enabled || !policy.emit_tool_records() {
        return Ok(());
    }

    start_request_trace_tool_event_ingest(
        drt,
        local_model,
        policy.tool_events_zmq_endpoint.clone(),
        policy.tool_events_zmq_topic.clone(),
    )
    .await
}

pub fn publish(record: RequestTraceRecord) {
    BUS.publish_ref(&record);
    if record.event_type == RequestTraceEventType::RequestPayload {
        PAYLOAD_BUS.publish(record);
    } else {
        METADATA_BUS.publish(record);
    }
}

pub fn subscribe() -> tokio::sync::broadcast::Receiver<RequestTraceRecord> {
    BUS.subscribe()
}

fn subscribe_sink() -> RequestTraceReceiver {
    RequestTraceReceiver {
        metadata: METADATA_BUS.subscribe(),
        payload: PAYLOAD_BUS.subscribe(),
    }
}

#[cfg(test)]
pub(crate) fn init_bus_for_test(capacity: usize) {
    BUS.init(capacity);
    METADATA_BUS.init(capacity);
    PAYLOAD_BUS.init(capacity);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(event_type: RequestTraceEventType) -> RequestTraceRecord {
        RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type,
            event_time_unix_ms: 0,
            event_source: None,
            agent_context: None,
            request: None,
            tool: None,
            payload: None,
        }
    }

    #[tokio::test]
    async fn payload_burst_does_not_evict_request_metadata() {
        init_bus_for_test(8);
        let mut receiver = subscribe_sink();

        publish(record(RequestTraceEventType::RequestEnd));
        for _ in 0..64 {
            publish(record(RequestTraceEventType::RequestPayload));
        }

        assert_eq!(
            receiver.metadata.recv().await.unwrap().event_type,
            RequestTraceEventType::RequestEnd
        );
    }
}
